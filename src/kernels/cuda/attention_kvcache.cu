#include "cuda/cuda_common.h"
#include "cuda/cuda_attention_kvcache.h"
#define WARP_SIZE 32
#define BLOCKSIZE WARP_SIZE
#define SEQ_UNIT 16

// ASSUME SEQ_LEN OF Q IS 1
__global__ void _attention_kvcache_kernel_128_1(float* input_k_cache,
                                              float* input_v_cache, 
                                              float* input_q, 
                                              float* input_k, 
                                              float* input_v, 
                                              int* position_id,
                                              AttentionKVCacheMetadata compMeta,
                                              float* output_O_temp,
                                              float* output_sum_temp) {
    int seq_length = position_id[0] + 1;
    int stride = (seq_length + SEQ_UNIT - 1) / SEQ_UNIT;
    if(blockIdx.y >= stride)
        return;

    int lane_id = threadIdx.x % WARP_SIZE;
    int group_id = threadIdx.x / WARP_SIZE;
    int parallel_idx = blockIdx.x * (blockDim.x / WARP_SIZE) + group_id;
    int idx_seq = blockIdx.y * SEQ_UNIT; 

    if(parallel_idx >= compMeta.dimSize[0] * compMeta.dimSize[1])
        return;

    float ptr_V[SEQ_UNIT*4]; // V
    float ptr_K[SEQ_UNIT*4]; // K
    float ptr_Q[4];          // Q
    float ptr_P[SEQ_UNIT] = {0};

    float ptr_O[4] = {0};
    float ptr_sum[1] = {0};

    // readin Q
    (float4 &)ptr_Q[0] = (float4 &)input_q[(lane_id * 4) + (parallel_idx * 128)];
    int common_idx = (lane_id * 4) + (parallel_idx * compMeta.stride[1]);

    // Q*K
    #pragma unroll
    for (int idx_SEQ_UNIT = 0; idx_SEQ_UNIT < SEQ_UNIT && idx_SEQ_UNIT + idx_seq < seq_length; idx_SEQ_UNIT ++) { 
        if(idx_SEQ_UNIT + idx_seq < seq_length - 1){                  
            (float4 &)ptr_K[idx_SEQ_UNIT * 4] 
                = (float4 &) input_k_cache[common_idx + ((idx_SEQ_UNIT + idx_seq) * compMeta.stride[2])];
        }
        else{
            (float4 &)ptr_K[idx_SEQ_UNIT * 4] 
                = (float4 &) input_k[((lane_id * 4) + parallel_idx * compMeta.stride[2])];
            (float4 &)input_k_cache[common_idx + ((idx_SEQ_UNIT + idx_seq) * compMeta.stride[2])] =
                (float4 &)ptr_K[idx_SEQ_UNIT * 4];
        }

        
        #pragma unroll
        for (int i = 0; i < 4; i ++){
            ptr_K[idx_SEQ_UNIT * 4 + i] = ptr_Q[i] * ptr_K[idx_SEQ_UNIT * 4 + i];
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                ptr_K[idx_SEQ_UNIT * 4 + i] += __shfl_down_sync(0xffffffff, ptr_K[idx_SEQ_UNIT * 4 + i], offset);
            }
            ptr_P[idx_SEQ_UNIT] += ptr_K[idx_SEQ_UNIT * 4 + i];
        }
    }

    // div sqrt(d)
    #pragma unroll
    for (int idx_SEQ_UNIT = 0; idx_SEQ_UNIT < SEQ_UNIT && idx_SEQ_UNIT + idx_seq < seq_length; idx_SEQ_UNIT ++) { 
        ptr_P[idx_SEQ_UNIT] = __shfl_sync(0xffffffff, ptr_P[idx_SEQ_UNIT], 0); 
        ptr_P[idx_SEQ_UNIT] /= sqrt(128.0);
    }

    // softmax
    #pragma unroll
    for (int idx_SEQ_UNIT = 0; idx_SEQ_UNIT < SEQ_UNIT && idx_SEQ_UNIT + idx_seq < seq_length; idx_SEQ_UNIT ++) { 
        ptr_P[idx_SEQ_UNIT] = expf(ptr_P[idx_SEQ_UNIT]);
        ptr_sum[0] += ptr_P[idx_SEQ_UNIT];
    }

    // * V
    #pragma unroll
    for (int idx_SEQ_UNIT = 0; idx_SEQ_UNIT < SEQ_UNIT && idx_SEQ_UNIT + idx_seq < seq_length; idx_SEQ_UNIT ++) { 
        if(idx_SEQ_UNIT + idx_seq < seq_length - 1){                  
            (float4 &)ptr_V[idx_SEQ_UNIT * 4] 
                = (float4 &) input_v_cache[common_idx + ((idx_SEQ_UNIT + idx_seq) * compMeta.stride[2])];
        }
        else{
            (float4 &)ptr_V[idx_SEQ_UNIT * 4] 
                = (float4 &) input_v[((lane_id * 4) + parallel_idx * compMeta.stride[2])];
            (float4 &)input_v_cache[common_idx + ((idx_SEQ_UNIT + idx_seq) * compMeta.stride[2])]
                = (float4 &)ptr_V[idx_SEQ_UNIT * 4];
        }

        #pragma unroll
        for (int i = 0; i < 4; i ++)
            ptr_O[i] = fmaf(ptr_P[idx_SEQ_UNIT], ptr_V[(idx_SEQ_UNIT * 4 + i)], ptr_O[i]);
    }

    #pragma unroll
    for (int i = 0; i < 4; i ++)
        ptr_O[i] /= ptr_sum[0];

    (float4 &)output_O_temp[(lane_id * 4) + (blockIdx.y * compMeta.dimSize[3]) + (parallel_idx * compMeta.dimSize[3] * stride)] = (float4 &)ptr_O[0];
    if(lane_id == 0){
        output_sum_temp[blockIdx.y + parallel_idx * stride] = ptr_sum[0];
    }

}

__global__ void _attention_kvcache_kernel_128_2(int* position_id,
                                              float* output_matmul,
                                              AttentionKVCacheMetadata compMeta,
                                              float* output_O_temp,
                                              float* output_sum_temp) {
    int lane_id = threadIdx.x % WARP_SIZE;
    int group_id = threadIdx.x / WARP_SIZE;
    int parallel_idx = blockIdx.x * (blockDim.x / WARP_SIZE) + group_id;

    float ptr_O[4] = {0};
    float ptr_O_sum[4] = {0};
    float ptr_sum = 0;
    float ptr_sum_temp;
    int size = (position_id[0] + SEQ_UNIT) / SEQ_UNIT;

    #pragma unroll
    for(int i = 0; i < size; i ++){
        (float4 &)ptr_O[0] 
            = (float4 &)output_O_temp[(lane_id * 4) + (i * compMeta.dimSize[3]) + parallel_idx * compMeta.dimSize[3] * size];
        ptr_sum_temp = output_sum_temp[i + parallel_idx * size];

        #pragma unroll
        for(int k = 0; k < 4; k ++)
            ptr_O_sum[k] += ptr_O[k] * ptr_sum_temp;
        ptr_sum += ptr_sum_temp;
    }

    #pragma unroll
    for(int k = 0; k < 4; k ++)
        ptr_O_sum[k] = ptr_O_sum[k] / ptr_sum; 

    (float4 &)output_matmul[(lane_id * 4) + (parallel_idx * compMeta.dimSize[3])] = (float4 &)ptr_O_sum[0];

}


namespace infini {
void attention_kvcache_kernel(float *input_k_cache, float *input_v_cache, 
                          float *input_q, float *input_k,
                          float *input_v, int *position_id, float *output_matmul, 
                          const AttentionKVCacheMetadata &compMeta,
                          float *output_O_temp, float *output_sum_temp) {
    IT_ASSERT(compMeta.dimSize[3] == 128);

    int gridsize_y = (compMeta.dimSize[2] - 1 + SEQ_UNIT) / SEQ_UNIT;
    dim3 gridDim(compMeta.dimSize[0]*compMeta.dimSize[1]/(BLOCKSIZE/WARP_SIZE), gridsize_y);
    dim3 blockDim(BLOCKSIZE, 1);

    _attention_kvcache_kernel_128_1
        <<<gridDim, blockDim, 0, CUDAStream::getCurrentStream()>>>
        (input_k_cache, input_v_cache, input_q, input_k, input_v, position_id,
        compMeta, output_O_temp, output_sum_temp);

    _attention_kvcache_kernel_128_2
        <<<compMeta.dimSize[0]*compMeta.dimSize[1]/(BLOCKSIZE/WARP_SIZE), WARP_SIZE,
        0, CUDAStream::getCurrentStream()>>>
        (position_id, output_matmul, compMeta, output_O_temp, output_sum_temp);
}

} // namespace infini

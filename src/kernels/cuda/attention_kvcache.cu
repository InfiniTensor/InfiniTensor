#include "cuda/cuda_common.h"
#include "cuda/cuda_attention_kvcache.h"
#define WARP_SIZE 32
#define BLOCKSIZE WARP_SIZE*2
#define SEQ_UNIT 8

// ASSUME SEQ_LEN OF Q IS 1
__global__ void _attention_kvcache_kernel_128_1(float* input_k_cache,
                                              float* input_v_cache, 
                                              float* input_q, 
                                              float* input_k, 
                                              float* input_v, 
                                              int* position_id,
                                              AttentionKVCacheMetadata compMeta,
                                              half* output_O_temp,
                                              float* output_sum_temp) {
    int seq_length = position_id[0] + 1;
    int stride = (seq_length + SEQ_UNIT - 1) / SEQ_UNIT;
    if(blockIdx.y >= stride)
        return;

    int lane_id_x4 = threadIdx.x % WARP_SIZE * 4;
    int group_id = threadIdx.x / WARP_SIZE;
    int parallel_idx = blockIdx.x * (blockDim.x / WARP_SIZE) + group_id;
    int idx_seq = blockIdx.y * SEQ_UNIT; 

    if(parallel_idx >= compMeta.dimSize[0] * compMeta.dimSize[1])
        return;

    half ptr_V[4]; // V
    half ptr_K[4]; // K
    half ptr_Q[4];          // Q
    float ptr_P[SEQ_UNIT] = {0};

    float ptr_O[4] = {0};
    float ptr_sum[1] = {0};
    float temp[4];

    // readin Q
    (float4 &)temp[0] = (float4 &)input_q[lane_id_x4 + (parallel_idx * 128)];
    for(int i = 0; i < 4; i += 2){
        *((half2*)(&ptr_Q[i])) = __float22half2_rn(*((float2*)(&temp[i])));
    }
    int common_idx = lane_id_x4 + (parallel_idx * compMeta.stride[1]);
    int idx_kv = lane_id_x4 + parallel_idx * compMeta.stride[2];
    
    // Q*K
    #pragma unroll
    for (int idx_SEQ_UNIT = 0; idx_SEQ_UNIT < SEQ_UNIT && idx_SEQ_UNIT + idx_seq < seq_length; idx_SEQ_UNIT ++) { 
        int idx_kvcache = common_idx + ((idx_SEQ_UNIT + idx_seq) * compMeta.stride[2]);
        if(idx_SEQ_UNIT + idx_seq < seq_length - 1){          
            *((int2*)(&ptr_K[0])) = *((int2*)(&((half*)input_k_cache)[idx_kvcache]));       
        }
        else{
            (float4 &)temp[0] = (float4 &) input_k[idx_kv];
            for(int i = 0; i < 4; i += 2)
                *((half2*)(&ptr_K[i])) = __float22half2_rn(*((float2*)(&temp[i])));
            *((int2*)(&((half*)input_k_cache)[idx_kvcache])) = *((int2*)(&ptr_K[0]));
        }
        // * V
        if(idx_SEQ_UNIT + idx_seq < seq_length - 1){                 
            *((int2*)(&ptr_V[0])) = *((int2*)(&((half*)input_v_cache)[idx_kvcache]));  
        }
        else{
            (float4 &)temp[0] = (float4 &) input_v[idx_kv];
            for(int i = 0; i < 4; i += 2)
                *((half2*)(&ptr_V[i])) = __float22half2_rn(*((float2*)(&temp[i])));
            *((int2*)(&((half*)input_v_cache)[idx_kvcache])) = *((int2*)(&ptr_V[0]));
        }
        
        #pragma unroll
        for (int i = 0; i < 4; i ++){
            ptr_K[i] = ptr_Q[i] * ptr_K[i];
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                ptr_K[i] += __shfl_down_sync(0xffffffff, ptr_K[i], offset);
            }
            ptr_P[idx_SEQ_UNIT] += __half2float(ptr_K[i]);
        }

        // div sqrt(d)
        ptr_P[idx_SEQ_UNIT] = __shfl_sync(0xffffffff, ptr_P[idx_SEQ_UNIT], 0); 
        ptr_P[idx_SEQ_UNIT] /= sqrt(128.0);

        // softmax
        ptr_P[idx_SEQ_UNIT] = expf(ptr_P[idx_SEQ_UNIT]);
        ptr_sum[0] += ptr_P[idx_SEQ_UNIT];

        #pragma unroll
        for (int i = 0; i < 4; i ++)
            ptr_O[i] = fmaf(ptr_P[idx_SEQ_UNIT], __half2float(ptr_V[i]), ptr_O[i]);
    }

    #pragma unroll
    for (int i = 0; i < 4; i ++)
        ptr_O[i] /= ptr_sum[0];

    for(int i = 0; i < 4; i += 2)
        (half2 &)output_O_temp[(lane_id_x4 + i) + (blockIdx.y * compMeta.dimSize[3]) + (parallel_idx * compMeta.dimSize[3] * stride)] = __float22half2_rn((float2 &)ptr_O[i]);
    if(lane_id_x4 == 0){
        output_sum_temp[blockIdx.y + parallel_idx * stride] = ptr_sum[0];
    }

}

__global__ void _attention_kvcache_kernel_128_2(int* position_id,
                                              float* output_matmul,
                                              AttentionKVCacheMetadata compMeta,
                                              half* output_O_temp,
                                              float* output_sum_temp) {
    int lane_id_x2 = threadIdx.x % WARP_SIZE * 2;
    int group_id = threadIdx.x / WARP_SIZE;
    int parallel_idx = blockIdx.x * (blockDim.x / WARP_SIZE) + group_id;

    float ptr_O[4] = {0};
    float ptr_O_sum[4] = {0};
    float ptr_sum = 0;
    float ptr_sum_temp;
    int size = (position_id[0] + SEQ_UNIT) / SEQ_UNIT;

    if(size > 1){
        #pragma unroll
        for(int i = 0; i < size; i ++){
            for(int j = 0; j < 4; j += 2)
                (float2 &)ptr_O[j] 
                    = __half22float2((half2 &)output_O_temp[(lane_id_x2 + j*32) + (i * compMeta.dimSize[3]) + parallel_idx * compMeta.dimSize[3] * size]);
            ptr_sum_temp = output_sum_temp[i + parallel_idx * size];

            #pragma unroll
            for(int k = 0; k < 4; k ++)
                ptr_O_sum[k] += ptr_O[k] * ptr_sum_temp;
            ptr_sum += ptr_sum_temp;
        }

        #pragma unroll
        for(int k = 0; k < 4; k ++)
            ptr_O_sum[k] = ptr_O_sum[k] / ptr_sum; 
        
        for(int j = 0; j < 4; j += 2)
            (float2 &)output_matmul[lane_id_x2 + j*32 + (parallel_idx * compMeta.dimSize[3])] = (float2 &)ptr_O_sum[j];
    }
    else{
        for(int i = 0; i < 4; i += 2)
            (float2 &)output_matmul[(lane_id_x2 + i*32) + (parallel_idx * compMeta.dimSize[3])] 
                = __half22float2((half2 &)output_O_temp[(lane_id_x2 + i*32) + parallel_idx * compMeta.dimSize[3]]);
    }
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
        compMeta, (half*)output_O_temp, output_sum_temp);

    _attention_kvcache_kernel_128_2
        <<<compMeta.dimSize[0]*compMeta.dimSize[1], WARP_SIZE,
        0, CUDAStream::getCurrentStream()>>>
        (position_id, output_matmul, compMeta, (half*)output_O_temp, output_sum_temp);
}

} // namespace infini

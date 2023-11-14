#include "cuda/cuda_common.h"
#include "cuda/cuda_attention_kvcache.h"
#define WARP_SIZE 32
#define BLOCKSIZE WARP_SIZE
#define SEQ_UNIT 64

__global__ void _attention_kvcache_kernel(float* input_k_cache,
                                              float* input_v_cache, 
                                              float* input_q, 
                                              float* input_k, 
                                              float* input_v, 
                                              int* position_id,
                                              float* output_matmul,
                                              AttentionKVCacheMetadata compMeta) {
    int lane_id = threadIdx.x % WARP_SIZE;
    int group_id = threadIdx.x / WARP_SIZE;
    int parallel_idx = blockIdx.x * (blockDim.x / WARP_SIZE) + group_id;

    if(parallel_idx >= compMeta.dimSize[0] * compMeta.dimSize[1])
        return;

    float ptr_V[SEQ_UNIT*2]; 
    float ptr_K[SEQ_UNIT*2]; 
    float ptr_Q[2]; 
    float ptr_P[SEQ_UNIT];

    float ptr_O[2];
    float ptr_max[1];
    float ptr_sum[1];

    float ptr_max_last[1];
    float ptr_sum_last[1];
    float ptr_O_last[2];

    (float2 &)ptr_Q[0] = (float2 &)input_q[(lane_id * 2) + (parallel_idx * 64)];

    int SEQ_LENGTH = position_id[0] + 1;

    int common_idx = (lane_id * 2) + (parallel_idx * compMeta.stride[1]);


    for (int idx_seq = 0; idx_seq < SEQ_LENGTH; idx_seq += SEQ_UNIT){ 
        ptr_max_last[0] = ptr_max[0];
        ptr_sum_last[0] = ptr_sum[0];
        (float2 &)ptr_O_last[0] = (float2 &)ptr_O[0];

        #pragma unroll
        for (int idx_SEQ_UNIT = 0; idx_SEQ_UNIT < SEQ_UNIT && idx_SEQ_UNIT + idx_seq < SEQ_LENGTH; idx_SEQ_UNIT ++) { 
            if(idx_SEQ_UNIT + idx_seq < SEQ_LENGTH - 1){                  
                (float2 &)ptr_K[idx_SEQ_UNIT * 2] 
                    = (float2 &) input_k_cache[common_idx + ((idx_SEQ_UNIT + idx_seq) * compMeta.stride[2])];
            }
            else{
                (float2 &)ptr_K[idx_SEQ_UNIT * 2] 
                    = (float2 &) input_k[((lane_id * 2) + parallel_idx * compMeta.stride[2])];
                (float2 &)input_k_cache[common_idx + ((idx_SEQ_UNIT + idx_seq) * compMeta.stride[2])] =
                    (float2 &)ptr_K[idx_SEQ_UNIT * 2];
            }
            ptr_K[idx_SEQ_UNIT * 2] = ptr_Q[0] * ptr_K[idx_SEQ_UNIT * 2];
            ptr_K[idx_SEQ_UNIT * 2 + 1] = ptr_Q[1] * ptr_K[idx_SEQ_UNIT * 2 + 1];

            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                ptr_K[idx_SEQ_UNIT * 2] += __shfl_down_sync(0xffffffff, ptr_K[idx_SEQ_UNIT * 2], offset);
            }
            ptr_P[idx_SEQ_UNIT] = ptr_K[idx_SEQ_UNIT * 2];
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2){
                ptr_K[((idx_SEQ_UNIT * 2) + 1)] += __shfl_down_sync(0xffffffff, ptr_K[((idx_SEQ_UNIT * 2) + 1)], offset);
            }
            ptr_P[idx_SEQ_UNIT] += ptr_K[((idx_SEQ_UNIT * 2) + 1)];
        }

        #pragma unroll
        for (int idx_SEQ_UNIT = 0; idx_SEQ_UNIT < SEQ_UNIT && idx_SEQ_UNIT + idx_seq < SEQ_LENGTH; idx_SEQ_UNIT ++) { 
            ptr_P[idx_SEQ_UNIT] = __shfl_sync(0xffffffff, ptr_P[idx_SEQ_UNIT], 0); 
            ptr_P[idx_SEQ_UNIT] /= 8;
            ptr_max[0] = (idx_SEQ_UNIT == 0) ? ptr_P[0] : max(ptr_max[0], ptr_P[idx_SEQ_UNIT]);
        }
        ptr_max[0] = (idx_seq == 0) ? ptr_max[0] : max(ptr_max[0], ptr_max_last[0]);

        ptr_sum[0] = 0;
        #pragma unroll
        for (int idx_SEQ_UNIT = 0; idx_SEQ_UNIT < SEQ_UNIT && idx_SEQ_UNIT + idx_seq < SEQ_LENGTH; idx_SEQ_UNIT ++) { 
            ptr_P[idx_SEQ_UNIT] = expf(ptr_P[idx_SEQ_UNIT] - ptr_max[0]);
            ptr_sum[0] += ptr_P[idx_SEQ_UNIT];
        }
        ptr_sum[0] = (idx_seq == 0) ? ptr_sum[0] : expf(ptr_max_last[0] - ptr_max[0]) * ptr_sum_last[0] + ptr_sum[0]; 

        ptr_O[0] = 0;
        ptr_O[1] = 0;
        #pragma unroll
        for (int idx_SEQ_UNIT = 0; idx_SEQ_UNIT < SEQ_UNIT && idx_SEQ_UNIT + idx_seq < SEQ_LENGTH; idx_SEQ_UNIT ++) { 
            if(idx_SEQ_UNIT + idx_seq < SEQ_LENGTH - 1){                  
                (float2 &)ptr_V[idx_SEQ_UNIT * 2] 
                    = (float2 &) input_v_cache[common_idx + ((idx_SEQ_UNIT + idx_seq) * compMeta.stride[2])];
            }
            else{
                (float2 &)ptr_V[idx_SEQ_UNIT * 2] 
                    = (float2 &) input_v[((lane_id * 2) + parallel_idx * compMeta.stride[2])];
                (float2 &)input_v_cache[common_idx + ((idx_SEQ_UNIT + idx_seq) * compMeta.stride[2])] =
                    (float2 &)ptr_V[idx_SEQ_UNIT * 2];
            }

            ptr_P[idx_SEQ_UNIT] /= ptr_sum[0];

            ptr_O[0] = fmaf(ptr_P[idx_SEQ_UNIT],  ptr_V[(idx_SEQ_UNIT * 2)], ptr_O[0]);
            ptr_O[1] = fmaf(ptr_P[idx_SEQ_UNIT],  ptr_V[(idx_SEQ_UNIT * 2) + 1], ptr_O[1]);
        }
        ptr_O[0] = (idx_seq == 0) ? ptr_O[0] : ptr_O[0] + ptr_O_last[0]  * expf(ptr_max_last[0] - ptr_max[0]) * ptr_sum_last[0] / ptr_sum[0];  
        ptr_O[1] = (idx_seq == 0) ? ptr_O[1] : ptr_O[1] + ptr_O_last[1]  * expf(ptr_max_last[0] - ptr_max[0]) * ptr_sum_last[0] / ptr_sum[0]; 
    }
    (float2 &)output_matmul[(lane_id * 2) + (parallel_idx * compMeta.dimSize[3])] = (float2 &)ptr_O[0];
}

namespace infini {
void attention_kvcache_kernel(float *input_k_cache, float *input_v_cache, float *input_q, float *input_k,
                          float *input_v, int *position_id, float *output_matmul,
                          const AttentionKVCacheMetadata &compMeta) {
    IT_ASSERT(compMeta.dimSize[3] == 64);
    dim3 gridDim(compMeta.dimSize[0]*compMeta.dimSize[1]/(BLOCKSIZE/WARP_SIZE), 1);
    dim3 blockDim(BLOCKSIZE, 1);

    _attention_kvcache_kernel<<<gridDim, blockDim>>>(
        input_k_cache, input_v_cache, input_q, input_k, input_v, position_id, output_matmul, compMeta);
}

} // namespace infini

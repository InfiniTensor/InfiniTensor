#include "cuda/cuda_common.h"
#include "cuda/cuda_attention_kvcache.h"
#define WARP_SIZE 32
#define BLOCKSIZE WARP_SIZE
#define SEQ_UNIT 32

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

__global__ void _attention_kvcache_kernel_128(float* input_k_cache,
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

    float ptr_V[SEQ_UNIT*4]; 
    float ptr_K[SEQ_UNIT*4]; 
    float ptr_Q[4]; 
    float ptr_P[SEQ_UNIT];

    float ptr_O[4];
    float ptr_max[1];
    float ptr_sum[1];

    float ptr_max_last[1];
    float ptr_sum_last[1];
    float ptr_O_last[4];

    (float4 &)ptr_Q[0] = (float4 &)input_q[(lane_id * 4) + (parallel_idx * 128)];

    int SEQ_LENGTH = position_id[0] + 1;

    int common_idx = (lane_id * 4) + (parallel_idx * compMeta.stride[1]);


    for (int idx_seq = 0; idx_seq < SEQ_LENGTH; idx_seq += SEQ_UNIT){ 
        ptr_max_last[0] = ptr_max[0];
        ptr_sum_last[0] = ptr_sum[0];
        (float4 &)ptr_O_last[0] = (float4 &)ptr_O[0];

        #pragma unroll
        for (int idx_SEQ_UNIT = 0; idx_SEQ_UNIT < SEQ_UNIT && idx_SEQ_UNIT + idx_seq < SEQ_LENGTH; idx_SEQ_UNIT ++) { 
            if(idx_SEQ_UNIT + idx_seq < SEQ_LENGTH - 1){                  
                (float4 &)ptr_K[idx_SEQ_UNIT * 4] 
                    = (float4 &) input_k_cache[common_idx + ((idx_SEQ_UNIT + idx_seq) * compMeta.stride[2])];
            }
            else{
                (float4 &)ptr_K[idx_SEQ_UNIT * 4] 
                    = (float4 &) input_k[((lane_id * 4) + parallel_idx * compMeta.stride[2])];
                (float4 &)input_k_cache[common_idx + ((idx_SEQ_UNIT + idx_seq) * compMeta.stride[2])] =
                    (float4 &)ptr_K[idx_SEQ_UNIT * 4];
            }
            ptr_K[idx_SEQ_UNIT * 4] = ptr_Q[0] * ptr_K[idx_SEQ_UNIT * 4];
            ptr_K[idx_SEQ_UNIT * 4 + 1] = ptr_Q[1] * ptr_K[idx_SEQ_UNIT * 4 + 1];
            ptr_K[idx_SEQ_UNIT * 4 + 2] = ptr_Q[2] * ptr_K[idx_SEQ_UNIT * 4 + 2];
            ptr_K[idx_SEQ_UNIT * 4 + 3] = ptr_Q[3] * ptr_K[idx_SEQ_UNIT * 4 + 3];

            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                ptr_K[idx_SEQ_UNIT * 4] += __shfl_down_sync(0xffffffff, ptr_K[idx_SEQ_UNIT * 4], offset);
            }
            ptr_P[idx_SEQ_UNIT] = ptr_K[idx_SEQ_UNIT * 4];

            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2){
                ptr_K[((idx_SEQ_UNIT * 4) + 1)] += __shfl_down_sync(0xffffffff, ptr_K[((idx_SEQ_UNIT * 4) + 1)], offset);
            }
            ptr_P[idx_SEQ_UNIT] += ptr_K[((idx_SEQ_UNIT * 4) + 1)];

            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2){
                ptr_K[((idx_SEQ_UNIT * 4) + 2)] += __shfl_down_sync(0xffffffff, ptr_K[((idx_SEQ_UNIT * 4) + 2)], offset);
            }
            ptr_P[idx_SEQ_UNIT] += ptr_K[((idx_SEQ_UNIT * 4) + 2)];

            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2){
                ptr_K[((idx_SEQ_UNIT * 4) + 3)] += __shfl_down_sync(0xffffffff, ptr_K[((idx_SEQ_UNIT * 4) + 3)], offset);
            }
            ptr_P[idx_SEQ_UNIT] += ptr_K[((idx_SEQ_UNIT * 4) + 3)];
        }

        #pragma unroll
        for (int idx_SEQ_UNIT = 0; idx_SEQ_UNIT < SEQ_UNIT && idx_SEQ_UNIT + idx_seq < SEQ_LENGTH; idx_SEQ_UNIT ++) { 
            ptr_P[idx_SEQ_UNIT] = __shfl_sync(0xffffffff, ptr_P[idx_SEQ_UNIT], 0); 
            ptr_P[idx_SEQ_UNIT] /= sqrt(128.0);
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
        ptr_O[2] = 0;
        ptr_O[3] = 0;
        #pragma unroll
        for (int idx_SEQ_UNIT = 0; idx_SEQ_UNIT < SEQ_UNIT && idx_SEQ_UNIT + idx_seq < SEQ_LENGTH; idx_SEQ_UNIT ++) { 
            if(idx_SEQ_UNIT + idx_seq < SEQ_LENGTH - 1){                  
                (float4 &)ptr_V[idx_SEQ_UNIT * 4] 
                    = (float4 &) input_v_cache[common_idx + ((idx_SEQ_UNIT + idx_seq) * compMeta.stride[2])];
            }
            else{
                (float4 &)ptr_V[idx_SEQ_UNIT * 4] 
                    = (float4 &) input_v[((lane_id * 4) + parallel_idx * compMeta.stride[2])];
                (float4 &)input_v_cache[common_idx + ((idx_SEQ_UNIT + idx_seq) * compMeta.stride[2])]
                    = (float4 &)ptr_V[idx_SEQ_UNIT * 4];
            }

            ptr_P[idx_SEQ_UNIT] /= ptr_sum[0];

            ptr_O[0] = fmaf(ptr_P[idx_SEQ_UNIT],  ptr_V[(idx_SEQ_UNIT * 4)], ptr_O[0]);
            ptr_O[1] = fmaf(ptr_P[idx_SEQ_UNIT],  ptr_V[(idx_SEQ_UNIT * 4) + 1], ptr_O[1]);
            ptr_O[2] = fmaf(ptr_P[idx_SEQ_UNIT],  ptr_V[(idx_SEQ_UNIT * 4) + 2], ptr_O[2]);
            ptr_O[3] = fmaf(ptr_P[idx_SEQ_UNIT],  ptr_V[(idx_SEQ_UNIT * 4) + 3], ptr_O[3]);
        }
        ptr_O[0] = (idx_seq == 0) ? ptr_O[0] : ptr_O[0] + ptr_O_last[0]  * expf(ptr_max_last[0] - ptr_max[0]) * ptr_sum_last[0] / ptr_sum[0];  
        ptr_O[1] = (idx_seq == 0) ? ptr_O[1] : ptr_O[1] + ptr_O_last[1]  * expf(ptr_max_last[0] - ptr_max[0]) * ptr_sum_last[0] / ptr_sum[0]; 
        ptr_O[2] = (idx_seq == 0) ? ptr_O[2] : ptr_O[2] + ptr_O_last[2]  * expf(ptr_max_last[0] - ptr_max[0]) * ptr_sum_last[0] / ptr_sum[0];  
        ptr_O[3] = (idx_seq == 0) ? ptr_O[3] : ptr_O[3] + ptr_O_last[3]  * expf(ptr_max_last[0] - ptr_max[0]) * ptr_sum_last[0] / ptr_sum[0]; 
    }
    (float4 &)output_matmul[(lane_id * 4) + (parallel_idx * compMeta.dimSize[3])] = (float4 &)ptr_O[0];
}

__global__ void _attention_kvcache_kernel_128_sum_only(float* input_k_cache,
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

    float ptr_V[SEQ_UNIT*4]; 
    float ptr_K[SEQ_UNIT*4]; 
    float ptr_Q[4]; 
    float ptr_P[SEQ_UNIT];

    float ptr_O[4];
    float ptr_sum[1];

    float ptr_sum_last[1];
    float ptr_O_last[4];

    (float4 &)ptr_Q[0] = (float4 &)input_q[(lane_id * 4) + (parallel_idx * 128)];

    int SEQ_LENGTH = position_id[0] + 1;

    int common_idx = (lane_id * 4) + (parallel_idx * compMeta.stride[1]);

    for (int idx_seq = 0; idx_seq < SEQ_LENGTH; idx_seq += SEQ_UNIT){ 
        ptr_sum_last[0] = ptr_sum[0];
        (float4 &)ptr_O_last[0] = (float4 &)ptr_O[0];

        #pragma unroll
        for (int idx_SEQ_UNIT = 0; idx_SEQ_UNIT < SEQ_UNIT && idx_SEQ_UNIT + idx_seq < SEQ_LENGTH; idx_SEQ_UNIT ++) { 
            if(idx_SEQ_UNIT + idx_seq < SEQ_LENGTH - 1){                  
                (float4 &)ptr_K[idx_SEQ_UNIT * 4] 
                    = (float4 &) input_k_cache[common_idx + ((idx_SEQ_UNIT + idx_seq) * compMeta.stride[2])];
            }
            else{
                (float4 &)ptr_K[idx_SEQ_UNIT * 4] 
                    = (float4 &) input_k[((lane_id * 4) + parallel_idx * compMeta.stride[2])];
                (float4 &)input_k_cache[common_idx + ((idx_SEQ_UNIT + idx_seq) * compMeta.stride[2])] =
                    (float4 &)ptr_K[idx_SEQ_UNIT * 4];
            }
            ptr_K[idx_SEQ_UNIT * 4] = ptr_Q[0] * ptr_K[idx_SEQ_UNIT * 4];
            ptr_K[idx_SEQ_UNIT * 4 + 1] = ptr_Q[1] * ptr_K[idx_SEQ_UNIT * 4 + 1];
            ptr_K[idx_SEQ_UNIT * 4 + 2] = ptr_Q[2] * ptr_K[idx_SEQ_UNIT * 4 + 2];
            ptr_K[idx_SEQ_UNIT * 4 + 3] = ptr_Q[3] * ptr_K[idx_SEQ_UNIT * 4 + 3];

            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                ptr_K[idx_SEQ_UNIT * 4] += __shfl_down_sync(0xffffffff, ptr_K[idx_SEQ_UNIT * 4], offset);
            }
            ptr_P[idx_SEQ_UNIT] = ptr_K[idx_SEQ_UNIT * 4];

            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2){
                ptr_K[((idx_SEQ_UNIT * 4) + 1)] += __shfl_down_sync(0xffffffff, ptr_K[((idx_SEQ_UNIT * 4) + 1)], offset);
            }
            ptr_P[idx_SEQ_UNIT] += ptr_K[((idx_SEQ_UNIT * 4) + 1)];

            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2){
                ptr_K[((idx_SEQ_UNIT * 4) + 2)] += __shfl_down_sync(0xffffffff, ptr_K[((idx_SEQ_UNIT * 4) + 2)], offset);
            }
            ptr_P[idx_SEQ_UNIT] += ptr_K[((idx_SEQ_UNIT * 4) + 2)];

            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2){
                ptr_K[((idx_SEQ_UNIT * 4) + 3)] += __shfl_down_sync(0xffffffff, ptr_K[((idx_SEQ_UNIT * 4) + 3)], offset);
            }
            ptr_P[idx_SEQ_UNIT] += ptr_K[((idx_SEQ_UNIT * 4) + 3)];
        }

        #pragma unroll
        for (int idx_SEQ_UNIT = 0; idx_SEQ_UNIT < SEQ_UNIT && idx_SEQ_UNIT + idx_seq < SEQ_LENGTH; idx_SEQ_UNIT ++) { 
            ptr_P[idx_SEQ_UNIT] = __shfl_sync(0xffffffff, ptr_P[idx_SEQ_UNIT], 0); 
            ptr_P[idx_SEQ_UNIT] /= sqrt(128.0);
        }

        ptr_sum[0] = 0;
        #pragma unroll
        for (int idx_SEQ_UNIT = 0; idx_SEQ_UNIT < SEQ_UNIT && idx_SEQ_UNIT + idx_seq < SEQ_LENGTH; idx_SEQ_UNIT ++) { 
            ptr_P[idx_SEQ_UNIT] = expf(ptr_P[idx_SEQ_UNIT]);
            ptr_sum[0] += ptr_P[idx_SEQ_UNIT];
        }
        ptr_sum[0] = (idx_seq == 0) ? ptr_sum[0] : ptr_sum_last[0] + ptr_sum[0]; 

        ptr_O[0] = 0;
        ptr_O[1] = 0;
        ptr_O[2] = 0;
        ptr_O[3] = 0;
        #pragma unroll
        for (int idx_SEQ_UNIT = 0; idx_SEQ_UNIT < SEQ_UNIT && idx_SEQ_UNIT + idx_seq < SEQ_LENGTH; idx_SEQ_UNIT ++) { 
            if(idx_SEQ_UNIT + idx_seq < SEQ_LENGTH - 1){                  
                (float4 &)ptr_V[idx_SEQ_UNIT * 4] 
                    = (float4 &) input_v_cache[common_idx + ((idx_SEQ_UNIT + idx_seq) * compMeta.stride[2])];
            }
            else{
                (float4 &)ptr_V[idx_SEQ_UNIT * 4] 
                    = (float4 &) input_v[((lane_id * 4) + parallel_idx * compMeta.stride[2])];
                (float4 &)input_v_cache[common_idx + ((idx_SEQ_UNIT + idx_seq) * compMeta.stride[2])]
                    = (float4 &)ptr_V[idx_SEQ_UNIT * 4];
            }

            ptr_P[idx_SEQ_UNIT] /= ptr_sum[0];

            ptr_O[0] = fmaf(ptr_P[idx_SEQ_UNIT],  ptr_V[(idx_SEQ_UNIT * 4)], ptr_O[0]);
            ptr_O[1] = fmaf(ptr_P[idx_SEQ_UNIT],  ptr_V[(idx_SEQ_UNIT * 4) + 1], ptr_O[1]);
            ptr_O[2] = fmaf(ptr_P[idx_SEQ_UNIT],  ptr_V[(idx_SEQ_UNIT * 4) + 2], ptr_O[2]);
            ptr_O[3] = fmaf(ptr_P[idx_SEQ_UNIT],  ptr_V[(idx_SEQ_UNIT * 4) + 3], ptr_O[3]);
        }
        ptr_O[0] = (idx_seq == 0) ? ptr_O[0] : ptr_O[0] + ptr_O_last[0]  * ptr_sum_last[0] / ptr_sum[0];  
        ptr_O[1] = (idx_seq == 0) ? ptr_O[1] : ptr_O[1] + ptr_O_last[1]  * ptr_sum_last[0] / ptr_sum[0]; 
        ptr_O[2] = (idx_seq == 0) ? ptr_O[2] : ptr_O[2] + ptr_O_last[2]  * ptr_sum_last[0] / ptr_sum[0];  
        ptr_O[3] = (idx_seq == 0) ? ptr_O[3] : ptr_O[3] + ptr_O_last[3]  * ptr_sum_last[0] / ptr_sum[0]; 
    }
    (float4 &)output_matmul[(lane_id * 4) + (parallel_idx * compMeta.dimSize[3])] = (float4 &)ptr_O[0];
}

__global__ void _attention_kvcache_kernel_128_sum_only_cp(float* input_k_cache,
                                              float* input_v_cache, 
                                              float* input_q, 
                                              float* input_k, 
                                              float* input_v, 
                                              int position_id,
                                              float* output_matmul,
                                              AttentionKVCacheMetadata compMeta) {
    int lane_id = threadIdx.x % WARP_SIZE;
    int group_id = threadIdx.x / WARP_SIZE;
    int parallel_idx = blockIdx.x * (blockDim.x / WARP_SIZE) + group_id;

    if(parallel_idx >= compMeta.dimSize[0] * compMeta.dimSize[1])
        return;

    float ptr_V[SEQ_UNIT*4]; 
    float ptr_K[SEQ_UNIT*4]; 
    float ptr_Q[4]; 
    float ptr_P[SEQ_UNIT];

    float ptr_O[4];
    float ptr_sum[1] = {0};

    float ptr_O_last[4];

    (float4 &)ptr_Q[0] = (float4 &)input_q[(lane_id * 4) + (parallel_idx * 128)];

    int SEQ_LENGTH = position_id + 1;

    int common_idx = (lane_id * 4) + (parallel_idx * compMeta.stride[1]);

    for (int idx_seq = 0; idx_seq < SEQ_LENGTH; idx_seq += SEQ_UNIT){ 
        (float4 &)ptr_O_last[0] = (float4 &)ptr_O[0];
        #pragma unroll
        for (int idx_SEQ_UNIT = 0; idx_SEQ_UNIT < SEQ_UNIT && idx_SEQ_UNIT + idx_seq < SEQ_LENGTH; idx_SEQ_UNIT ++) { 
            if(idx_SEQ_UNIT + idx_seq < SEQ_LENGTH - 1){                  
                (float4 &)ptr_K[idx_SEQ_UNIT * 4] 
                    = (float4 &) input_k_cache[common_idx + ((idx_SEQ_UNIT + idx_seq) * compMeta.stride[2])];
            }
            else{
                (float4 &)ptr_K[idx_SEQ_UNIT * 4] 
                    = (float4 &) input_k[((lane_id * 4) + parallel_idx * compMeta.stride[2])];
                (float4 &)input_k_cache[common_idx + ((idx_SEQ_UNIT + idx_seq) * compMeta.stride[2])] =
                    (float4 &)ptr_K[idx_SEQ_UNIT * 4];
            }
            ptr_K[idx_SEQ_UNIT * 4] = ptr_Q[0] * ptr_K[idx_SEQ_UNIT * 4];
            ptr_K[idx_SEQ_UNIT * 4 + 1] = ptr_Q[1] * ptr_K[idx_SEQ_UNIT * 4 + 1];
            ptr_K[idx_SEQ_UNIT * 4 + 2] = ptr_Q[2] * ptr_K[idx_SEQ_UNIT * 4 + 2];
            ptr_K[idx_SEQ_UNIT * 4 + 3] = ptr_Q[3] * ptr_K[idx_SEQ_UNIT * 4 + 3];

            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                ptr_K[idx_SEQ_UNIT * 4] += __shfl_down_sync(0xffffffff, ptr_K[idx_SEQ_UNIT * 4], offset);
            }
            ptr_P[idx_SEQ_UNIT] = ptr_K[idx_SEQ_UNIT * 4];

            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2){
                ptr_K[((idx_SEQ_UNIT * 4) + 1)] += __shfl_down_sync(0xffffffff, ptr_K[((idx_SEQ_UNIT * 4) + 1)], offset);
            }
            ptr_P[idx_SEQ_UNIT] += ptr_K[((idx_SEQ_UNIT * 4) + 1)];

            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2){
                ptr_K[((idx_SEQ_UNIT * 4) + 2)] += __shfl_down_sync(0xffffffff, ptr_K[((idx_SEQ_UNIT * 4) + 2)], offset);
            }
            ptr_P[idx_SEQ_UNIT] += ptr_K[((idx_SEQ_UNIT * 4) + 2)];

            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2){
                ptr_K[((idx_SEQ_UNIT * 4) + 3)] += __shfl_down_sync(0xffffffff, ptr_K[((idx_SEQ_UNIT * 4) + 3)], offset);
            }
            ptr_P[idx_SEQ_UNIT] += ptr_K[((idx_SEQ_UNIT * 4) + 3)];
        }

        #pragma unroll
        for (int idx_SEQ_UNIT = 0; idx_SEQ_UNIT < SEQ_UNIT && idx_SEQ_UNIT + idx_seq < SEQ_LENGTH; idx_SEQ_UNIT ++) { 
            ptr_P[idx_SEQ_UNIT] = __shfl_sync(0xffffffff, ptr_P[idx_SEQ_UNIT], 0); 
            ptr_P[idx_SEQ_UNIT] /= sqrt(128.0);
        }

        #pragma unroll
        for (int idx_SEQ_UNIT = 0; idx_SEQ_UNIT < SEQ_UNIT && idx_SEQ_UNIT + idx_seq < SEQ_LENGTH; idx_SEQ_UNIT ++) { 
            ptr_P[idx_SEQ_UNIT] = expf(ptr_P[idx_SEQ_UNIT]);
            ptr_sum[0] += ptr_P[idx_SEQ_UNIT];
        }

        ptr_O[0] = 0;
        ptr_O[1] = 0;
        ptr_O[2] = 0;
        ptr_O[3] = 0;
        #pragma unroll
        for (int idx_SEQ_UNIT = 0; idx_SEQ_UNIT < SEQ_UNIT && idx_SEQ_UNIT + idx_seq < SEQ_LENGTH; idx_SEQ_UNIT ++) { 
            if(idx_SEQ_UNIT + idx_seq < SEQ_LENGTH - 1){                  
                (float4 &)ptr_V[idx_SEQ_UNIT * 4] 
                    = (float4 &) input_v_cache[common_idx + ((idx_SEQ_UNIT + idx_seq) * compMeta.stride[2])];
            }
            else{
                (float4 &)ptr_V[idx_SEQ_UNIT * 4] 
                    = (float4 &) input_v[((lane_id * 4) + parallel_idx * compMeta.stride[2])];
                (float4 &)input_v_cache[common_idx + ((idx_SEQ_UNIT + idx_seq) * compMeta.stride[2])]
                    = (float4 &)ptr_V[idx_SEQ_UNIT * 4];
            }

            ptr_O[0] = fmaf(ptr_P[idx_SEQ_UNIT],  ptr_V[(idx_SEQ_UNIT * 4)],     ptr_O[0]);
            ptr_O[1] = fmaf(ptr_P[idx_SEQ_UNIT],  ptr_V[(idx_SEQ_UNIT * 4) + 1], ptr_O[1]);
            ptr_O[2] = fmaf(ptr_P[idx_SEQ_UNIT],  ptr_V[(idx_SEQ_UNIT * 4) + 2], ptr_O[2]);
            ptr_O[3] = fmaf(ptr_P[idx_SEQ_UNIT],  ptr_V[(idx_SEQ_UNIT * 4) + 3], ptr_O[3]);
        }
        ptr_O[0] = (idx_seq == 0) ? ptr_O[0] : ptr_O[0] + ptr_O_last[0];  
        ptr_O[1] = (idx_seq == 0) ? ptr_O[1] : ptr_O[1] + ptr_O_last[1]; 
        ptr_O[2] = (idx_seq == 0) ? ptr_O[2] : ptr_O[2] + ptr_O_last[2];  
        ptr_O[3] = (idx_seq == 0) ? ptr_O[3] : ptr_O[3] + ptr_O_last[3]; 
    }
    ptr_O[0] = ptr_O[0] / ptr_sum[0];  
    ptr_O[1] = ptr_O[1] / ptr_sum[0]; 
    ptr_O[2] = ptr_O[2] / ptr_sum[0];  
    ptr_O[3] = ptr_O[3] / ptr_sum[0]; 

    (float4 &)output_matmul[(lane_id * 4) + (parallel_idx * compMeta.dimSize[3])] = (float4 &)ptr_O[0];
}

__global__ void _attention_kvcache_kernel_128_sum_only_1(float* input_k_cache,
                                              float* input_v_cache, 
                                              float* input_q, 
                                              float* input_k, 
                                              float* input_v, 
                                              int position_id,
                                              float* output_matmul,
                                              AttentionKVCacheMetadata compMeta,
                                              float* output_O_temp,
                                              float* output_sum_temp) {
    int lane_id = threadIdx.x % WARP_SIZE;
    int group_id = threadIdx.x / WARP_SIZE;
    int parallel_idx = blockIdx.x * (blockDim.x / WARP_SIZE) + group_id;
    int SEQ_LENGTH = position_id + 1;
    int idx_seq = blockIdx.y * SEQ_UNIT; 

    if(parallel_idx >= compMeta.dimSize[0] * compMeta.dimSize[1] && idx_seq >= SEQ_LENGTH)
        return;

    float ptr_V[SEQ_UNIT*4]; 
    float ptr_K[SEQ_UNIT*4]; 
    float ptr_Q[4]; 
    float ptr_P[SEQ_UNIT];

    float ptr_O[4];
    float ptr_sum[1] = {0};

    (float4 &)ptr_Q[0] = (float4 &)input_q[(lane_id * 4) + (parallel_idx * 128)];
    int common_idx = (lane_id * 4) + (parallel_idx * compMeta.stride[1]);

    #pragma unroll
    for (int idx_SEQ_UNIT = 0; idx_SEQ_UNIT < SEQ_UNIT && idx_SEQ_UNIT + idx_seq < SEQ_LENGTH; idx_SEQ_UNIT ++) { 
        if(idx_SEQ_UNIT + idx_seq < SEQ_LENGTH - 1){                  
            (float4 &)ptr_K[idx_SEQ_UNIT * 4] 
                = (float4 &) input_k_cache[common_idx + ((idx_SEQ_UNIT + idx_seq) * compMeta.stride[2])];
        }
        else{
            (float4 &)ptr_K[idx_SEQ_UNIT * 4] 
                = (float4 &) input_k[((lane_id * 4) + parallel_idx * compMeta.stride[2])];
            (float4 &)input_k_cache[common_idx + ((idx_SEQ_UNIT + idx_seq) * compMeta.stride[2])] =
                (float4 &)ptr_K[idx_SEQ_UNIT * 4];
        }
        ptr_K[idx_SEQ_UNIT * 4]     = ptr_Q[0] * ptr_K[idx_SEQ_UNIT * 4];
        ptr_K[idx_SEQ_UNIT * 4 + 1] = ptr_Q[1] * ptr_K[idx_SEQ_UNIT * 4 + 1];
        ptr_K[idx_SEQ_UNIT * 4 + 2] = ptr_Q[2] * ptr_K[idx_SEQ_UNIT * 4 + 2];
        ptr_K[idx_SEQ_UNIT * 4 + 3] = ptr_Q[3] * ptr_K[idx_SEQ_UNIT * 4 + 3];

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            ptr_K[idx_SEQ_UNIT * 4] += __shfl_down_sync(0xffffffff, ptr_K[idx_SEQ_UNIT * 4], offset);
        }
        ptr_P[idx_SEQ_UNIT] = ptr_K[idx_SEQ_UNIT * 4];

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2){
            ptr_K[((idx_SEQ_UNIT * 4) + 1)] += __shfl_down_sync(0xffffffff, ptr_K[((idx_SEQ_UNIT * 4) + 1)], offset);
        }
        ptr_P[idx_SEQ_UNIT] += ptr_K[((idx_SEQ_UNIT * 4) + 1)];

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2){
            ptr_K[((idx_SEQ_UNIT * 4) + 2)] += __shfl_down_sync(0xffffffff, ptr_K[((idx_SEQ_UNIT * 4) + 2)], offset);
        }
        ptr_P[idx_SEQ_UNIT] += ptr_K[((idx_SEQ_UNIT * 4) + 2)];

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2){
            ptr_K[((idx_SEQ_UNIT * 4) + 3)] += __shfl_down_sync(0xffffffff, ptr_K[((idx_SEQ_UNIT * 4) + 3)], offset);
        }
        ptr_P[idx_SEQ_UNIT] += ptr_K[((idx_SEQ_UNIT * 4) + 3)];
    }

    #pragma unroll
    for (int idx_SEQ_UNIT = 0; idx_SEQ_UNIT < SEQ_UNIT && idx_SEQ_UNIT + idx_seq < SEQ_LENGTH; idx_SEQ_UNIT ++) { 
        ptr_P[idx_SEQ_UNIT] = __shfl_sync(0xffffffff, ptr_P[idx_SEQ_UNIT], 0); 
        ptr_P[idx_SEQ_UNIT] /= sqrt(128.0);
    }

    #pragma unroll
    for (int idx_SEQ_UNIT = 0; idx_SEQ_UNIT < SEQ_UNIT && idx_SEQ_UNIT + idx_seq < SEQ_LENGTH; idx_SEQ_UNIT ++) { 
        ptr_P[idx_SEQ_UNIT] = expf(ptr_P[idx_SEQ_UNIT]);
        ptr_sum[0] += ptr_P[idx_SEQ_UNIT];
    }

    ptr_O[0] = 0;
    ptr_O[1] = 0;
    ptr_O[2] = 0;
    ptr_O[3] = 0;
    #pragma unroll
    for (int idx_SEQ_UNIT = 0; idx_SEQ_UNIT < SEQ_UNIT && idx_SEQ_UNIT + idx_seq < SEQ_LENGTH; idx_SEQ_UNIT ++) { 
        if(idx_SEQ_UNIT + idx_seq < SEQ_LENGTH - 1){                  
            (float4 &)ptr_V[idx_SEQ_UNIT * 4] 
                = (float4 &) input_v_cache[common_idx + ((idx_SEQ_UNIT + idx_seq) * compMeta.stride[2])];
        }
        else{
            (float4 &)ptr_V[idx_SEQ_UNIT * 4] 
                = (float4 &) input_v[((lane_id * 4) + parallel_idx * compMeta.stride[2])];
            (float4 &)input_v_cache[common_idx + ((idx_SEQ_UNIT + idx_seq) * compMeta.stride[2])]
                = (float4 &)ptr_V[idx_SEQ_UNIT * 4];
        }

        ptr_O[0] = fmaf(ptr_P[idx_SEQ_UNIT],  ptr_V[(idx_SEQ_UNIT * 4)],     ptr_O[0]);
        ptr_O[1] = fmaf(ptr_P[idx_SEQ_UNIT],  ptr_V[(idx_SEQ_UNIT * 4) + 1], ptr_O[1]);
        ptr_O[2] = fmaf(ptr_P[idx_SEQ_UNIT],  ptr_V[(idx_SEQ_UNIT * 4) + 2], ptr_O[2]);
        ptr_O[3] = fmaf(ptr_P[idx_SEQ_UNIT],  ptr_V[(idx_SEQ_UNIT * 4) + 3], ptr_O[3]);
    }

    if(gridDim.y == 1){
        ptr_O[0] /= ptr_sum[0];
        ptr_O[1] /= ptr_sum[0];
        ptr_O[2] /= ptr_sum[0];
        ptr_O[3] /= ptr_sum[0];
    }
    else if(threadIdx.x == 0){
        output_sum_temp[blockIdx.y + parallel_idx * gridDim.y] = ptr_sum[0];
    }

    (float4 &)output_O_temp[(lane_id * 4) + (blockIdx.y * compMeta.dimSize[3]) + (parallel_idx * compMeta.dimSize[3] * gridDim.y)] = (float4 &)ptr_O[0];
}

__global__ void _attention_kvcache_kernel_128_sum_only_2(float* input_k_cache,
                                              float* input_v_cache, 
                                              float* input_q, 
                                              float* input_k, 
                                              float* input_v, 
                                              int size,
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

    #pragma unroll
    for(int i = 0; i < size; i ++){
        (float4 &)ptr_O[0] 
            = (float4 &)output_O_temp[(lane_id * 4) + (i * compMeta.dimSize[3]) + parallel_idx * compMeta.dimSize[3] * size];
        ptr_O_sum[0] += ptr_O[0];
        ptr_O_sum[1] += ptr_O[1];
        ptr_O_sum[2] += ptr_O[2];
        ptr_O_sum[3] += ptr_O[3];
        ptr_sum += output_sum_temp[i + parallel_idx * size];
    }

    ptr_O_sum[0] = ptr_O_sum[0] / ptr_sum;  
    ptr_O_sum[1] = ptr_O_sum[1] / ptr_sum; 
    ptr_O_sum[2] = ptr_O_sum[2] / ptr_sum;  
    ptr_O_sum[3] = ptr_O_sum[3] / ptr_sum; 

    (float4 &)output_matmul[(lane_id * 4) + (parallel_idx * compMeta.dimSize[3])] = (float4 &)ptr_O_sum[0];

}

namespace infini {
void attention_kvcache_kernel(float *input_k_cache, float *input_v_cache, float *input_q, float *input_k,
                          float *input_v, int *position_id, float *output_matmul,
                          const AttentionKVCacheMetadata &compMeta) {
    IT_ASSERT(compMeta.dimSize[3] == 64 || compMeta.dimSize[3] == 128);
    int position_id_h;
    cudaMemcpy(&position_id_h, position_id, sizeof(int), cudaMemcpyDeviceToHost);
    

    int gridsize_y = (position_id_h + SEQ_UNIT) / SEQ_UNIT;
    dim3 gridDim(compMeta.dimSize[0]*compMeta.dimSize[1]/(BLOCKSIZE/WARP_SIZE), gridsize_y);
    dim3 blockDim(BLOCKSIZE, 1);
    bool needReduce = gridsize_y > 1 ? true : false;
    float *output_O_temp, *output_sum_temp;
    if(needReduce){
        cudaMalloc((void **)&output_O_temp,   compMeta.dimSize[0]*compMeta.dimSize[1]*gridsize_y*WARP_SIZE*sizeof(float4));
        cudaMalloc((void **)&output_sum_temp, compMeta.dimSize[0]*compMeta.dimSize[1]*sizeof(float)*gridsize_y);
    }

    if(compMeta.dimSize[3] == 64)
        _attention_kvcache_kernel<<<gridDim, blockDim>>>(
            input_k_cache, input_v_cache, input_q, input_k, input_v, position_id, output_matmul, compMeta);
    else{
        //IT_ASSERT(compMeta.dimSize[0]*compMeta.dimSize[1]*gridsize_y*WARP_SIZE*sizeof(float4) < 16777216);
        if(!needReduce){
            _attention_kvcache_kernel_128_sum_only_1<<<gridDim, blockDim>>>(
                input_k_cache, input_v_cache, input_q, input_k, input_v, position_id_h, nullptr, compMeta, output_matmul, nullptr);
            // cudaDeviceSynchronize();
        }
        else{
            _attention_kvcache_kernel_128_sum_only_1<<<gridDim, blockDim>>>(
                input_k_cache, input_v_cache, input_q, input_k, input_v, position_id_h, nullptr, compMeta, output_O_temp, output_sum_temp);
            cudaDeviceSynchronize();
            //assert(gridsize_y < 33);
            _attention_kvcache_kernel_128_sum_only_2<<<compMeta.dimSize[0]*compMeta.dimSize[1]/(BLOCKSIZE/WARP_SIZE), WARP_SIZE>>>(
                input_k_cache, input_v_cache, input_q, input_k, input_v, gridsize_y, output_matmul, compMeta, output_O_temp, output_sum_temp);
            // cudaDeviceSynchronize();
        }
    }
}

} // namespace infini

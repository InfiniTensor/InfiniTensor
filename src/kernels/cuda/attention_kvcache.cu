#include "cuda/cuda_common.h"
#include "cuda/cuda_attention_kvcache.h"
#define WARP_SIZE 32
#define SEQ_UNIT 16
#define BLOCKSIZE_2 WARP_SIZE*4
#define MAX_PARTITION 1024

// ASSUME SEQ_LEN OF Q IS 1
template <class T>
__global__ void _attention_kvcache_kernel_128_1(T* input_k_cache,
                                              T* input_v_cache, 
                                              T* input_q, 
                                              T* input_k, 
                                              T* input_v, 
                                              int64_t* position_id,
                                              AttentionKVCacheMetadata compMeta,
                                              half* output_O_temp,
                                              float* output_sum_temp) {
    int seq_length = position_id[blockIdx.y] + 1;
    int stride = (seq_length + SEQ_UNIT - 1) / SEQ_UNIT;
    if(blockIdx.z >= stride)
        return;

    int lane_id_x2 = threadIdx.x % WARP_SIZE * 2;
    int parallel_idx = blockIdx.x + blockIdx.y * gridDim.x;

    int idx_seq = blockIdx.z * SEQ_UNIT;

    half reg_V[4];
    half reg_K[4];
    half reg_Q[4];
    float reg_P;

    float reg_O[4] = {0};
    float reg_sum = 0;
    float temp[4];
    bool is_fp16 = sizeof(T) == 2 ? true : false;

    int idx_qkv = lane_id_x2 + parallel_idx * compMeta.head_dim;

    // readin Q
    if(!is_fp16){
        #pragma unroll
        for(int i = 0; i < 4; i += 2){
            (float2 &)temp[i] = (float2 &)input_q[idx_qkv + i*WARP_SIZE];
            *((half2*)(&reg_Q[i])) = __float22half2_rn(*((float2*)(&temp[i])));
        }
    }
    else{
        #pragma unroll
        for(int i = 0; i < 4; i += 2){
            (half2 &)reg_Q[i] = (half2 &)input_q[idx_qkv + i*WARP_SIZE];
        }
    }
    int common_idx = lane_id_x2 + (parallel_idx * compMeta.max_kv_seqlen * compMeta.head_dim);
    
    #pragma unroll
    for (int idx_SEQ_UNIT = 0; idx_SEQ_UNIT < SEQ_UNIT && idx_SEQ_UNIT + idx_seq < seq_length; idx_SEQ_UNIT ++) { 
        reg_P = 0;
        int idx_kvcache = common_idx + ((idx_SEQ_UNIT + idx_seq) * compMeta.head_dim);
        // readin K & V
        if(idx_SEQ_UNIT + idx_seq < seq_length - 1){
            #pragma unroll
            for(int i = 0; i < 4; i += 2){
                *((half2*)(&reg_K[i])) = *((half2*)(&((half*)input_k_cache)[idx_kvcache + i*WARP_SIZE]));
                *((half2*)(&reg_V[i])) = *((half2*)(&((half*)input_v_cache)[idx_kvcache + i*WARP_SIZE]));
            }
        }
        else{
            if(!is_fp16){
                #pragma unroll
                for(int i = 0; i < 4; i += 2){
                    (float2 &)temp[i] = (float2 &) input_k[idx_qkv + i*WARP_SIZE];
                    *((half2*)(&reg_K[i])) = __float22half2_rn(*((float2*)(&temp[i])));
                    *((half2*)(&((half*)input_k_cache)[idx_kvcache + i*WARP_SIZE])) = *((half2*)(&reg_K[i]));
                    (float2 &)temp[i] = (float2 &) input_v[idx_qkv + i*WARP_SIZE];
                    *((half2*)(&reg_V[i])) = __float22half2_rn(*((float2*)(&temp[i])));
                    *((half2*)(&((half*)input_v_cache)[idx_kvcache + i*WARP_SIZE])) = *((half2*)(&reg_V[i]));
                }
            }
            else{
                #pragma unroll
                for(int i = 0; i < 4; i += 2){
                    (half2 &)reg_K[i] = (half2 &)input_k[idx_qkv + i*WARP_SIZE];
                    *((half2*)(&((half*)input_k_cache)[idx_kvcache + i*WARP_SIZE])) = *((half2*)(&reg_K[i]));
                    (half2 &)reg_V[i] = (half2 &)input_v[idx_qkv + i*WARP_SIZE];
                    *((half2*)(&((half*)input_v_cache)[idx_kvcache + i*WARP_SIZE])) = *((half2*)(&reg_V[i]));
                }
            }
        }

        // Q*K
        #pragma unroll
        for (int i = 0; i < 4; i += 2){
            (half2 &)reg_K[i] = (half2 &)reg_Q[i] * (half2 &)reg_K[i];
            #pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                (half2 &)reg_K[i] += __shfl_xor_sync(0xffffffff, (half2 &)reg_K[i], offset);
            }
            (float2 &) temp[i] = __half22float2((half2 &)reg_K[i]);
            reg_P += (temp[i] + temp[i+1]);
            (float2 &) temp[i] = __half22float2((half2 &)reg_V[i]);
        }

        // div sqrt(d)
        reg_P /= sqrt(128.0);

        // softmax
        reg_P = expf(reg_P);
        reg_sum += reg_P;

        #pragma unroll
        for (int i = 0; i < 4; i ++)
            reg_O[i] = fmaf(reg_P, temp[i], reg_O[i]);
    }

    #pragma unroll
    for (int i = 0; i < 4; i ++)
        reg_O[i] /= reg_sum;

    #pragma unroll
    for(int i = 0; i < 4; i += 2)
        (half2 &)output_O_temp[(lane_id_x2 + i*WARP_SIZE) + (blockIdx.z * compMeta.head_dim) + (parallel_idx * compMeta.head_dim * stride)] = __float22half2_rn((float2 &)reg_O[i]);
    if(lane_id_x2 == 0){
        output_sum_temp[blockIdx.z + parallel_idx * stride] = reg_sum;
    }
}


template <class T>
__global__ void _attention_kvcache_kernel_128_2(int64_t* position_id,
                                              T* output_matmul,
                                              AttentionKVCacheMetadata compMeta,
                                              half* output_O_temp,
                                              float* output_sum_temp) {
    int lane_id = threadIdx.x % WARP_SIZE;
    int parallel_idx = blockIdx.x;
    int offset = parallel_idx * compMeta.head_dim;


    int size = (position_id[0] + SEQ_UNIT) / SEQ_UNIT;
    bool is_fp16 = sizeof(T) == 2 ? true : false;

    if(size == 1){
        if(!is_fp16){
            #pragma unroll
            for(int i = threadIdx.x; i < compMeta.head_dim; i += blockDim.x)
                output_matmul[i + offset]
                    = __half2float(output_O_temp[i + offset]);
        }
        else{
            #pragma unroll
            for(int i = threadIdx.x; i < compMeta.head_dim; i += blockDim.x)
                output_matmul[i + offset]
                    = output_O_temp[i + offset];
        }
        return;
    }

    __shared__ float shm_sum_temp[MAX_PARTITION];
    __shared__ float shm_sum[WARP_SIZE];
    float temp_sum = 0;

    #pragma unroll
    for(int i = threadIdx.x; i < size; i += blockDim.x){
        shm_sum_temp[i] = output_sum_temp[i + parallel_idx * size];
        temp_sum += shm_sum_temp[i];
    }

    #pragma unroll
    for(int offset = WARP_SIZE/2; offset > 0; offset /= 2)
        temp_sum += __shfl_down_sync(0xffffffff, temp_sum, offset);
    if(lane_id == 0)
        shm_sum[threadIdx.x/WARP_SIZE] = temp_sum;
    __syncthreads();
    temp_sum = lane_id < (size + WARP_SIZE - 1) / WARP_SIZE ? shm_sum[lane_id] : 0;

    #pragma unroll
    for(int offset = WARP_SIZE/2; offset > 0; offset /= 2)
        temp_sum += __shfl_xor_sync(0xffffffff, temp_sum, offset);
    temp_sum = __fdividef(1.0f, temp_sum + 1e-6f);

    #pragma unroll
    for(int i = threadIdx.x; i < compMeta.head_dim; i += blockDim.x){
        float acc = 0.0f;
        for(int j = 0; j < size; j ++){
            acc = fma(__half2float(output_O_temp[i + (j * compMeta.head_dim) + offset * size]) * shm_sum_temp[j], temp_sum, acc);
        }

        if(!is_fp16){
            output_matmul[i + offset] = acc;
        }
        else{
            output_matmul[i + offset] = __float2half(acc);
        }
    }
}


namespace infini {
void attention_kvcache_kernel(int dType, void *input_k_cache, void *input_v_cache, 
                          void *input_q, void *input_k,
                          void *input_v, int64_t *position_id, void *output_matmul,
                          const AttentionKVCacheMetadata &compMeta,
                          float *output_O_temp, float *output_sum_temp) {
    IT_ASSERT(dType == 1 || dType == 10);

    int gridsize_y = (compMeta.max_kv_seqlen - 1 + SEQ_UNIT) / SEQ_UNIT;
    dim3 gridDim(compMeta.num_heads, compMeta.num_seqs, gridsize_y);
    dim3 blockDim(WARP_SIZE, 1);

    if(dType == 1){
        _attention_kvcache_kernel_128_1<float>
            <<<gridDim, blockDim, 0, CUDAStream::getCurrentStream()>>>
            ((float*)input_k_cache, (float*)input_v_cache, (float*)input_q, (float*)input_k, (float*)input_v, 
            position_id, compMeta, (half*)output_O_temp, output_sum_temp);

        _attention_kvcache_kernel_128_2<float>
            <<<compMeta.num_seqs*compMeta.num_heads, BLOCKSIZE_2,
            0, CUDAStream::getCurrentStream()>>>
            (position_id, (float*)output_matmul, compMeta, (half*)output_O_temp, output_sum_temp);
    }
    else{
        _attention_kvcache_kernel_128_1<half>
            <<<gridDim, blockDim, 0, CUDAStream::getCurrentStream()>>>
            ((half*)input_k_cache, (half*)input_v_cache, (half*)input_q, (half*)input_k, (half*)input_v, 
            position_id, compMeta, (half*)output_O_temp, output_sum_temp);

        _attention_kvcache_kernel_128_2<half>
            <<<compMeta.num_seqs*compMeta.num_heads, BLOCKSIZE_2,
            0, CUDAStream::getCurrentStream()>>>
            (position_id, (half*)output_matmul, compMeta, (half*)output_O_temp, output_sum_temp);
    }

}

} // namespace infini

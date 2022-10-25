#include "cuda_utils.h"
// Kernel
__global__ void kernel_func_2(float *tensor_ptr_9, float *tensor_ptr_10) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int parallel_idx = blockIdx.x * 8 + warp_id;
    float buf[8];
    for (int loop_idx = parallel_idx; loop_idx < 1024; loop_idx += 864) {
        int offset_src = 0;
        int tmp_offset_src = loop_idx;
        offset_src += tmp_offset_src % 1024 * 256;
        tmp_offset_src /= 1024;
        int offset_dst = 0;
        int tmp_offset_dst = loop_idx;
        offset_dst += tmp_offset_dst % 1024 * 256;
        tmp_offset_dst /= 1024;
#pragma unroll
        for (int inst_idx = 0; inst_idx < 8; inst_idx++) {
            buf[inst_idx] = tensor_ptr_9[0 + offset + inst_idx * 32 + lane_id];
        }
#pragma unroll
        for (int inst_idx = 0; inst_idx < 8; inst_idx++) {
            buf[inst_idx] = (buf[inst_idx] > 0) ? buf[inst_idx] : 0;
        }
#pragma unroll
        for (int inst_idx = 0; inst_idx < 8; inst_idx++) {
            tensor_ptr_10[0 + offset + inst_idx * 32 + lane_id] = buf[inst_idx];
        }
    }
}
// Kernel
__global__ void kernel_func_3(float *tensor_ptr_9, float *tensor_ptr_10,
                              float *tensor_ptr_11) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int parallel_idx = blockIdx.x * 8 + warp_id;
    float buf[24];
    for (int loop_idx = parallel_idx; loop_idx < 1024; loop_idx += 864) {
        int offset_src = 0;
        int tmp_offset_src = loop_idx;
        offset_src += tmp_offset_src % 1024 * 256;
        tmp_offset_src /= 1024;
        int offset_dst = 0;
        int tmp_offset_dst = loop_idx;
        offset_dst += tmp_offset_dst % 1024 * 256;
        tmp_offset_dst /= 1024;
#pragma unroll
        for (int inst_idx = 0; inst_idx < 8; inst_idx++) {
            buf[inst_idx] = tensor_ptr_9[0 + offset + inst_idx * 32 + lane_id];
        }
#pragma unroll
        for (int inst_idx = 0; inst_idx < 8; inst_idx++) {
            buf[inst_idx + 8] =
                tensor_ptr_10[0 + offset + inst_idx * 32 + lane_id];
        }
#pragma unroll
        for (int inst_idx = 0; inst_idx < 8; inst_idx++) {
            buf[inst_idx + 16] = buf[inst_idx] + buf[inst_idx + 8]
        }
#pragma unroll
        for (int inst_idx = 0; inst_idx < 8; inst_idx++) {
            tensor_ptr_11[0 + offset + inst_idx * 32 + lane_id] =
                buf[inst_idx + 16];
        }
    }
}
void invoke_func_2(float *src, float *dst) {
    dim3 gridDim(108, 1);
    dim3 blockDim(256, 1);
    kernel_func<<<gridDim, blockDim>>>(src, dst);
    cudaCheckError();
}
void invoke_func_3(float *src, float *dst) {
    dim3 gridDim(108, 1);
    dim3 blockDim(256, 1);
    kernel_func<<<gridDim, blockDim>>>(src, dst);
    cudaCheckError();
}

#include "cuda_utils.h"
// Kernel
__global__ void kernel_func_0(float *tensor_ptr_3, float *tensor_ptr_2,
                              float *tensor_ptr_4) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int parallel_idx = blockIdx.x * 2 + warp_id;
    float buf[24];
    for (int loop_idx = parallel_idx; loop_idx < 15627264; loop_idx += 216) {
        int offset_src = 0;
        int offset_src_buf = loop_idx;
        offset_src += offset_src_buf % 15627264 * 32;
        offset_src_buf /= 15627264;
    }
}
// Kernel
__global__ void kernel_func_6(float *tensor_ptr_4, float *tensor_ptr_5,
                              float *tensor_ptr_6, float *tensor_ptr_7,
                              float *tensor_ptr_8) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int parallel_idx = blockIdx.x * 8 + warp_id;
    float buf[48];
    for (int loop_idx = parallel_idx; loop_idx < 256; loop_idx += 864) {
        int offset_src = 0;
        int offset_src_buf = loop_idx;
        offset_src += offset_src_buf % 256 * 256;
        offset_src_buf /= 256;
#pragma unroll
        for (int inst_idx = 0; inst_idx < 8; inst_idx++) {
            buf[inst_idx] =
                tensor_ptr_4[0 + offset_src + inst_idx * 32 + lane_id];
        }
#pragma unroll
        for (int inst_idx = 0; inst_idx < 8; inst_idx++) {
            buf[inst_idx + 8] =
                tensor_ptr_5[0 + offset_src + inst_idx * 32 + lane_id];
        }
#pragma unroll
        for (int inst_idx = 0; inst_idx < 8; inst_idx++) {
            buf[inst_idx + 16] = buf[inst_idx] + buf[inst_idx + 8];
        }
#pragma unroll
        for (int inst_idx = 0; inst_idx < 8; inst_idx++) {
            tensor_ptr_6[0 + offset_src + inst_idx * 32 + lane_id] =
                buf[inst_idx + 16];
        }
#pragma unroll
        for (int inst_idx = 0; inst_idx < 8; inst_idx++) {
            buf[inst_idx] =
                tensor_ptr_6[0 + offset_src + inst_idx * 32 + lane_id];
        }
#pragma unroll
        for (int inst_idx = 0; inst_idx < 8; inst_idx++) {
            buf[inst_idx + 8] =
                tensor_ptr_7[0 + offset_src + inst_idx * 32 + lane_id];
        }
#pragma unroll
        for (int inst_idx = 0; inst_idx < 8; inst_idx++) {
            buf[inst_idx + 16] = buf[inst_idx] + buf[inst_idx + 8];
        }
#pragma unroll
        for (int inst_idx = 0; inst_idx < 8; inst_idx++) {
            tensor_ptr_8[0 + offset_src + inst_idx * 32 + lane_id] =
                buf[inst_idx + 16];
        }
    }
}
// Kernel
__global__ void kernel_func_7(float *tensor_ptr_8, float *tensor_ptr_9,
                              float *tensor_ptr_10) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int parallel_idx = blockIdx.x * 2 + warp_id;
    float buf[48];
    for (int loop_idx = parallel_idx; loop_idx < 128; loop_idx += 216) {
        int offset_src = 0;
        int offset_src_buf = loop_idx;
        offset_src += offset_src_buf % 128 * 512;
        offset_src_buf /= 128;
    }
}
// Kernel
__global__ void kernel_func_5(float *tensor_ptr_10, float *tensor_ptr_11,
                              float *tensor_ptr_12) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int parallel_idx = blockIdx.x * 8 + warp_id;
    float buf[24];
    for (int loop_idx = parallel_idx; loop_idx < 256; loop_idx += 864) {
        int offset_src = 0;
        int offset_src_buf = loop_idx;
        offset_src += offset_src_buf % 256 * 256;
        offset_src_buf /= 256;
#pragma unroll
        for (int inst_idx = 0; inst_idx < 8; inst_idx++) {
            buf[inst_idx] =
                tensor_ptr_10[0 + offset_src + inst_idx * 32 + lane_id];
        }
#pragma unroll
        for (int inst_idx = 0; inst_idx < 8; inst_idx++) {
            buf[inst_idx + 8] =
                tensor_ptr_11[0 + offset_src + inst_idx * 32 + lane_id];
        }
#pragma unroll
        for (int inst_idx = 0; inst_idx < 8; inst_idx++) {
            buf[inst_idx + 16] = buf[inst_idx] - buf[inst_idx + 8];
        }
#pragma unroll
        for (int inst_idx = 0; inst_idx < 8; inst_idx++) {
            tensor_ptr_12[0 + offset_src + inst_idx * 32 + lane_id] =
                buf[inst_idx + 16];
        }
    }
}
void invoke_func_0(float *tensor_ptr_3, float *tensor_ptr_2,
                   float *tensor_ptr_4) {
    dim3 gridDim(108, 1);
    dim3 blockDim(64, 1);
    kernel_func_0<<<gridDim, blockDim>>>(tensor_ptr_3, tensor_ptr_2,
                                         tensor_ptr_4);
    cudaCheckError();
}
void invoke_func_6(float *tensor_ptr_4, float *tensor_ptr_5,
                   float *tensor_ptr_6, float *tensor_ptr_7,
                   float *tensor_ptr_8) {
    dim3 gridDim(108, 1);
    dim3 blockDim(256, 1);
    kernel_func_6<<<gridDim, blockDim>>>(
        tensor_ptr_4, tensor_ptr_5, tensor_ptr_6, tensor_ptr_7, tensor_ptr_8);
    cudaCheckError();
}
void invoke_func_7(float *tensor_ptr_8, float *tensor_ptr_9,
                   float *tensor_ptr_10) {
    dim3 gridDim(108, 1);
    dim3 blockDim(64, 1);
    kernel_func_7<<<gridDim, blockDim>>>(tensor_ptr_8, tensor_ptr_9,
                                         tensor_ptr_10);
    cudaCheckError();
}
void invoke_func_5(float *tensor_ptr_10, float *tensor_ptr_11,
                   float *tensor_ptr_12) {
    dim3 gridDim(108, 1);
    dim3 blockDim(256, 1);
    kernel_func_5<<<gridDim, blockDim>>>(tensor_ptr_10, tensor_ptr_11,
                                         tensor_ptr_12);
    cudaCheckError();
}

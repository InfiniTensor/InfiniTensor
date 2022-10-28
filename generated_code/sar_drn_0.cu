#include "cuda_utils.h"
// Kernel
__global__ void kernel_func_2(float *tensor_ptr_2, float *tensor_ptr_4,
                              float *tensor_ptr_5) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int parallel_idx = blockIdx.x * 8 + warp_id;
    float buf[32];
    for (int loop_idx = parallel_idx; loop_idx < 65536; loop_idx += 864) {
        int offset_src = 0;
        int offset_src_buf = loop_idx;
        offset_src += offset_src_buf % 65536 * 256;
        offset_src_buf /= 65536;
#pragma unroll
        for (int inst_idx = 0; inst_idx < 8; inst_idx++) {
            buf[inst_idx] =
                tensor_ptr_2[0 + offset_src + inst_idx * 32 + lane_id];
        }
#pragma unroll
        for (int inst_idx = 0; inst_idx < 8; inst_idx++) {
            buf[inst_idx] = (buf[inst_idx] > 0) ? buf[inst_idx] : 0;
        }
#pragma unroll
        for (int inst_idx = 0; inst_idx < 8; inst_idx++) {
            buf[inst_idx + 8] =
                tensor_ptr_4[0 + offset_src + inst_idx * 32 + lane_id];
        }
#pragma unroll
        for (int inst_idx = 0; inst_idx < 8; inst_idx++) {
            buf[inst_idx + 16] = buf[inst_idx] + buf[inst_idx + 8];
        }
#pragma unroll
        for (int inst_idx = 0; inst_idx < 8; inst_idx++) {
            tensor_ptr_5[0 + offset_src + inst_idx * 32 + lane_id] =
                buf[inst_idx + 16];
        }
    }
}
void invoke_func_2(float *tensor_ptr_2, float *tensor_ptr_4,
                   float *tensor_ptr_5) {
    dim3 gridDim(108, 1);
    dim3 blockDim(256, 1);
    kernel_func_2<<<gridDim, blockDim>>>(tensor_ptr_2, tensor_ptr_4,
                                         tensor_ptr_5);
    cudaCheckError();
}

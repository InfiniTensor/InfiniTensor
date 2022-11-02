#include "cuda_utils.h"
// Kernel
__global__ void kernel_func_0(float *tensor_ptr_2, float *tensor_ptr_3) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int parallel_idx = blockIdx.x * 2 + warp_id;
    float buf[24];
    for (int loop_idx = parallel_idx; loop_idx < 128; loop_idx += 216) {
        int offset_src = 0;
        int offset_src_buf = loop_idx;
        offset_src += offset_src_buf % 128 * 512;
        offset_src_buf /= 128;
    }
}
void invoke_func_0(float *tensor_ptr_2, float *tensor_ptr_3) {
    dim3 gridDim(108, 1);
    dim3 blockDim(64, 1);
    kernel_func_0<<<gridDim, blockDim>>>(tensor_ptr_2, tensor_ptr_3);
    cudaCheckError();
}

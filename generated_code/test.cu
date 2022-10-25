#include "cuda_utils.h"
// Kernel
__global__ void kernel_func(float *src, float *dst) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int parallel_idx = blockIdx.x * 0 + warp_id;
    float buf[0];
    __shared__ float smem[0];
    for (int loop_idx = parallel_idx; loop_idx < 1024; loop_idx += 0) {
        if (lane_id < 0) {
#pragma unroll
            for (int inst_idx = 0; inst_idx < 32; inst_idx++) {
                smem[warp_id * 32 * 32 * 2 + inst_idx * 32 + lane_id] =
                    inst_idx;
            }
        }
    }
}
void invoke_func(float *src, float *dst) {
    dim3 gridDim(0, 1);
    dim3 blockDim(0, 1);
    kernel_func<<<gridDim, blockDim>>>(src, dst);
    cudaCheckError();
}

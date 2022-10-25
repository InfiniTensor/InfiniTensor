#include "cuda_utils.h"
// Kernel
__global__ void kernel_func(float *src, float *dst) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int parallel_idx = blockIdx.x * 8 + warp_id;
    float buf[8];
    for (int loop_idx = parallel_idx; loop_idx < 65536; loop_idx += 864) {
        int offset_src = 0;
        int tmp_offset_src = loop_idx;
        offset_src += tmp_offset_src % 65536 * 256;
        tmp_offset_src /= 65536;
        int offset_dst = 0;
        int tmp_offset_dst = loop_idx;
        offset_dst += tmp_offset_dst % 65536 * 256;
        tmp_offset_dst /= 65536;
#pragma unroll
        for (int inst_idx = 0; inst_idx < 8; inst_idx++) {
            buf[inst_idx] = src[0 + offset + inst_idx * 32 + lane_id];
        }
#pragma unroll
        for (int inst_idx = 0; inst_idx < 8; inst_idx++) {
            buf[inst_idx] = (buf[inst_idx] > 0) ? buf[inst_idx] : 0;
        }
#pragma unroll
        for (int inst_idx = 0; inst_idx < 8; inst_idx++) {
            dst[0 + offset + inst_idx * 32 + lane_id] = buf[inst_idx];
        }
    }
}
void invoke_func(float *src, float *dst) {
    dim3 gridDim(108, 1);
    dim3 blockDim(256, 1);
    kernel_func<<<gridDim, blockDim>>>(src, dst);
    cudaCheckError();
}

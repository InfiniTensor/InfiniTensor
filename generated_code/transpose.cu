#include "cuda_utils.h"
// Kernel
__global__ void kernel_func(float *src, float *dst) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int parallel_idx = blockIdx.x * 8 + warp_id;
    float buf[32];
    __shared__ float smem[8448];
    for (int loop_idx = parallel_idx; loop_idx < 1056; loop_idx += 864) {
        int offset_src = 0;
        int tmp_offset_src = loop_idx;
        offset_src += tmp_offset_src % 32 * 32736;
        tmp_offset_src /= 32;
        offset_src += tmp_offset_src % 33 * 32;
        tmp_offset_src /= 33;
        int offset_dst = 0;
        int tmp_offset_dst = loop_idx;
        offset_dst += tmp_offset_dst % 32 * 992;
        tmp_offset_dst /= 32;
        offset_dst += tmp_offset_dst % 33 * 31744;
        tmp_offset_dst /= 33;
#pragma unroll
        for (int inst_idx = 0; inst_idx < 31; inst_idx++) {
            buf[inst_idx] = src[0 + offset_src + 0 + inst_idx * 1056 + lane_id];
        }
#pragma unroll
        for (int inst_idx = 0; inst_idx < 31; inst_idx++) {
            smem[warp_id * 32 * 33 + inst_idx * 33 + lane_id] = buf[inst_idx];
        }
        if (lane_id < 31) {
#pragma unroll
            for (int inst_idx = 0; inst_idx < 32; inst_idx++) {
                buf[inst_idx] =
                    smem[warp_id * 32 * 33 + lane_id * 33 + inst_idx];
            }
        }
        if (lane_id < 31) {
#pragma unroll
            for (int inst_idx = 0; inst_idx < 32; inst_idx++) {
                dst[0 + offset_dst + 0 + inst_idx * 31 + lane_id] =
                    buf[inst_idx];
            }
        }
    }
}
void invoke_func(float *src, float *dst) {
    dim3 gridDim(108, 1);
    dim3 blockDim(256, 1);
    kernel_func<<<gridDim, blockDim>>>(src, dst);
    cudaCheckError();
}

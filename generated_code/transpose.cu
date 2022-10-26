#include "cuda_utils.h"
// Kernel
__global__ void kernel_func_0(float *tensor_ptr_2, float *tensor_ptr_3) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int parallel_idx = blockIdx.x * 8 + warp_id;
    float buf[32];
    __shared__ float smem[8448];
    for (int loop_idx = parallel_idx; loop_idx < 1056; loop_idx += 864) {
        int offset_src = 0;
        int offset_src_buf = loop_idx;
        offset_src += offset_src_buf % 32 * 32736;
        offset_src_buf /= 32;
        offset_src += offset_src_buf % 33 * 32;
        offset_src_buf /= 33;
        int offset_dst = 0;
        int offset_dst_buf = loop_idx;
        offset_dst += offset_dst_buf % 32 * 992;
        offset_dst_buf /= 32;
        offset_dst += offset_dst_buf % 33 * 31744;
        offset_dst_buf /= 33;
#pragma unroll
        for (int inst_idx = 0; inst_idx < 31; inst_idx++) {
            buf[inst_idx] =
                tensor_ptr_2[0 + offset_src + 0 + inst_idx * 1056 + lane_id];
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
                tensor_ptr_3[0 + offset_dst + 0 + inst_idx * 31 + lane_id] =
                    buf[inst_idx];
            }
        }
    }
}
void invoke_func_0(float *tensor_ptr_2, float *tensor_ptr_3) {
    dim3 gridDim(108, 1);
    dim3 blockDim(256, 1);
    kernel_func_0<<<gridDim, blockDim>>>(tensor_ptr_2, tensor_ptr_3);
    cudaCheckError();
}

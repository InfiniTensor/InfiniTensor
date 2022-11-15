#include "cuda_utils.h"
// Kernel
__global__ void kernel_func_3(float *input, float *bias, float *output) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int parallel_idx = blockIdx.x * 4 + warp_id;
    float buf[4];
    for (int loop_idx = parallel_idx; loop_idx < 0; loop_idx += 320) {
        int offset_input = 0;
        int offset_input_buf = loop_idx;
        offset_input += offset_input_buf % 1 * 128;
        offset_input_buf /= 1;
        offset_input += offset_input_buf % 232 * 49;
        offset_input_buf /= 232;
        int offset_bias = 0;
        int offset_bias_buf = loop_idx;
        offset_bias += offset_bias_buf % 232 * 232;
        offset_bias_buf /= 232;
#pragma unroll
        for (int inst_idx = 0; inst_idx < 4; inst_idx++) {
            buf[inst_idx] = input[0 + offset_input + inst_idx * 32 + lane_id];
        }
#pragma unroll
        for (int inst_idx = 0; inst_idx < 1; inst_idx++) {
            buf[4] = bias[0 + offset_bias];
        }
#pragma unroll
        for (int inst_idx = 0; inst_idx < 4; inst_idx++) {
            buf[inst_idx] = buf[inst_idx] + buf[4];
        }
#pragma unroll
        for (int inst_idx = 0; inst_idx < 8; inst_idx++) {
            bias[0 + offset_input + inst_idx * 32 + lane_id] = buf[inst_idx];
        }
    }
}
void invoke_func_3(float *input, float *bias, float *output) {
    dim3 gridDim(80, 1);
    dim3 blockDim(128, 1);
    kernel_func_3<<<gridDim, blockDim>>>(input, bias, output);
    cudaCheckError();
}

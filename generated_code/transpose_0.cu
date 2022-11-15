#include "cuda_utils.h"
// Kernel
__global__ void kernel_func_0(float *input, float *output) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int parallel_idx = blockIdx.x * 4 + warp_id;
    float buf[4];
    for (int loop_idx = parallel_idx; loop_idx < 812; loop_idx += 320) {
        int offset_input = 0;
        int offset_input_buf = loop_idx;
        offset_input += offset_input_buf % 7 * 128;
        offset_input_buf /= 7;
        offset_input += offset_input_buf % 58 * 1568;
        offset_input_buf /= 58;
        offset_input += offset_input_buf % 2 * 784;
        offset_input_buf /= 2;
        int offset_output = 0;
        int offset_output_buf = loop_idx;
        offset_output += offset_output_buf % 7 * 128;
        offset_output_buf /= 7;
        offset_output += offset_output_buf % 58 * 784;
        offset_output_buf /= 58;
        offset_output += offset_output_buf % 2 * 45472;
        offset_output_buf /= 2;
        if (loop_idx % 7 == 6) {
            if (lane_id < 16) {
                buf[0] = input[0 + offset_input + 0 * 32 + lane_id];
            }
        } else {
#pragma unroll
            for (int inst_idx = 0; inst_idx < 4; inst_idx++) {
                buf[inst_idx] =
                    input[0 + offset_input + inst_idx * 32 + lane_id];
            }
        }
        // test
        if (loop_idx % 7 == 6) {
            if (lane_id < 16) {
                output[0 + offset_output + 0 * 32 + lane_id] = buf[0];
            }
        } else {
#pragma unroll
            for (int inst_idx = 0; inst_idx < 4; inst_idx++) {
                output[0 + offset_output + inst_idx * 32 + lane_id] =
                    buf[inst_idx];
            }
        }
        // test
    }
}
void invoke_func_0(float *input, float *output) {
    dim3 gridDim(80, 1);
    dim3 blockDim(128, 1);
    kernel_func_0<<<gridDim, blockDim>>>(input, output);
    cudaCheckError();
}

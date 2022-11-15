#include "cuda_utils.h"
// Kernel
__global__ void kernel_func_2(float *input, float *output) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int parallel_idx = blockIdx.x * 4 + warp_id;
    float buf[4];
    for (int loop_idx = parallel_idx; loop_idx < 464; loop_idx += 320) {
        int offset_input = 0;
        int offset_input_buf = loop_idx;
        offset_input += offset_input_buf % 232 * 98;
        offset_input_buf /= 232;
        offset_input += offset_input_buf % 2 * 49;
        offset_input_buf /= 2;
        int offset_output = 0;
        int offset_output_buf = loop_idx;
        offset_output += offset_output_buf % 232 * 49;
        offset_output_buf /= 232;
        offset_output += offset_output_buf % 2 * 11368;
        offset_output_buf /= 2;
        if (loop_idx % 1 == 0) {
#pragma unroll
            for (int inst_idx = 0; inst_idx < 1; inst_idx++) {
                buf[inst_idx] =
                    input[0 + offset_input + inst_idx * 32 + lane_id];
            }
            if (lane_id < 17) {
                buf[1] = input[0 + offset_input + 1 * 32 + lane_id];
            }
        } else {
#pragma unroll
            for (int inst_idx = 0; inst_idx < 2; inst_idx++) {
                buf[inst_idx] =
                    input[0 + offset_input + inst_idx * 32 + lane_id];
            }
        }
        // test
        if (loop_idx % 1 == 0) {
#pragma unroll
            for (int inst_idx = 0; inst_idx < 1; inst_idx++) {
                output[0 + offset_output + inst_idx * 32 + lane_id] =
                    buf[inst_idx];
            }
            if (lane_id < 17) {
                output[0 + offset_output + 1 * 32 + lane_id] = buf[1];
            }
        } else {
#pragma unroll
            for (int inst_idx = 0; inst_idx < 2; inst_idx++) {
                output[0 + offset_output + inst_idx * 32 + lane_id] =
                    buf[inst_idx];
            }
        }
        // test
    }
}
void invoke_func_2(float *input, float *output) {
    dim3 gridDim(80, 1);
    dim3 blockDim(128, 1);
    kernel_func_2<<<gridDim, blockDim>>>(input, output);
    cudaCheckError();
}

// #include "core/common.h"
// #include "core/constants.h"
#include "cuda/cuda_common.h"
#include <math.h>

constexpr unsigned int num_threads() { return 32 * 4; }
constexpr int thread_work_size() { return 4; }
constexpr int block_work_size() { return thread_work_size() * num_threads(); }

__global__ 
void _leaky_relu_kernel(float *input, float *output, float alphaValue, int size) {

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < size; i += stride) {
        
        output[i] = (input[i] > 0) ? input[i] : alphaValue * input[i];
    }
    // if (index < size) {
    //     float effective_alpha = isnan(alphaValue) ? 0.01f : alphaValue; // If alpha is NaN£¬then we take 0.01f
    //     output[index] = (input[index] > 0) ? input[index] : effective_alpha * input[index];
    // }

}

namespace infini {
void leaky_relu_kernel(float *input, float *output, float alphaValue, int size) {

    int blocksize = block_work_size();
    int gridsize = (size + blocksize - 1) / blocksize;
    _leaky_relu_kernel<<<gridsize, blocksize>>>(input, output, alphaValue, size);

}

}; // namespace infini
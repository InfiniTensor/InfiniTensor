#include "cuda/cuda_common.h"

__global__
void _elu_kernel(const float* input, float* output, int size, float alpha) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        float x = input[index];
        output[index] = (x >= 0) ? x : alpha * (exp(x) - 1);
    }
}

namespace infini {
void elu_kernel(const float* input, float* output, int size, float alpha) {
    int blocksize = 32 * 16;
    int gridsize = (size + blocksize - 1) / blocksize;
    _elu_kernel<<<gridsize, blocksize>>>(input, output, size, alpha);
}
} // namespace infini

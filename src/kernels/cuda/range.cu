#include "cuda/cuda_common.h"
#include <iostream>

__global__
void _range_kernel(const float start, const float limit, const float delta,  float *output, int size) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        output[index] = start + (index * delta);
    }
}

namespace infini {
void range_kernel(const float start, const float limit, const float delta,  float *output, int size) {
    int blocksize = 32 * 16;
    int gridsize = (size + blocksize - 1) / blocksize;
    _range_kernel<<<blocksize, gridsize>>>(start, limit, delta, output, size);
}
} // namespace infini

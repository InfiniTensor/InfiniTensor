#include "cuda/cuda_common.h"
//默认处理的都是1D向量
__global__ void _where_kernel(const float *input, const float *other,
                              const float *condition, float *output, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        output[index] = condition[index] ? input[index] : other[index];
    }
}

namespace infini {
void where_kernel(const float *input, const float *other,
                  const float *condition, float *output, int size) {
    int blocksize = 32 * 16;
    int gridsize = (size + blocksize - 1) / blocksize;
    _where_kernel<<<blocksize, gridsize>>>(input, other, condition, output,
                                           size);
}
} // namespace infini

#include "core/common.h"
#include "cuda/cuda_common.h"
#include <math.h>

constexpr unsigned int num_threads() { return 32 * 4; }
constexpr int thread_work_size() { return 4; }
constexpr int block_work_size() { return thread_work_size() * num_threads(); }

__global__ void _clip_kernel(float *input, float *output, int n, float minValue,
                             float maxValue) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        output[i] = (!isnan(minValue) && input[i] < minValue)
                        ? minValue
                        : (!isnan(maxValue) && input[i] > maxValue)
                        ? maxValue : input[i];
    }
}

namespace infini {
void clip_kernel(float *input, float *output, int num, float minValue,
                 float maxValue) {
    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    _clip_kernel<<<blocksize, gridsize>>>(input, output, num, minValue,
                                          maxValue);
}

} // namespace infini

#include "core/common.h"
#include "core/constants.h"
#include "cuda/cuda_common.h"
#include <math.h>

using infini::E_CONSTANT;
constexpr unsigned int num_threads() { return 32 * 4; }
constexpr int thread_work_size() { return 4; }
constexpr int block_work_size() { return thread_work_size() * num_threads(); }

template <typename T>
__global__ void _clip_kernel(T *input, T *output, int n, float minValue,
                             float maxValue) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        float value = static_cast<float>(input[i]);
        output[i] = (!isnan(minValue) && value < minValue)
                        ? minValue
                        : (!isnan(maxValue) && value > maxValue)
                        ? maxValue : value;
    }
}

namespace infini {
template <typename T>
void clip_impl(T *input, T *output, int num, float minValue,
                 float maxValue) {
    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    _clip_kernel<T>
        <<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(
        input, output, num, minValue, maxValue);
}
void clip_kernel(float *input, float *output, int num, float minValue,
                 float maxValue) {
    clip_impl(input, output, num, minValue, maxValue);
}
void clip_kernel(half *input, half *output, int num, float minValue,
                 float maxValue) {
    clip_impl(input, output, num, minValue, maxValue);
}

}; // namespace infini

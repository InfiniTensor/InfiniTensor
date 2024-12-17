#include "core/common.h"
#include "core/constants.h"
#include "cuda/cuda_clip.h"
#include "cuda/cuda_common.h"
#include <math.h>

using infini::E_CONSTANT;
constexpr unsigned int num_threads() { return 32 * 4; }
constexpr int thread_work_size() { return 4; }
constexpr int block_work_size() { return thread_work_size() * num_threads(); }

template <class T>
__global__ void _clip_kernel(T *input, T *output, int n, T *minValue,
                             T *maxValue) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        output[i] = (minValue && input[i] < *minValue)   ? *minValue
                    : (maxValue && input[i] > *maxValue) ? *maxValue
                                                         : input[i];
    }
}

namespace infini {
template <typename T>
void clip_kernel(T *input, T *output, int num, T *minValue, T *maxValue) {
    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    _clip_kernel<<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(
        input, output, num, minValue, maxValue);
    cudaDeviceSynchronize();
}

template void clip_kernel<float>(float *input, float *output, int num,
                                 float *minValue, float *maxValue);
template void clip_kernel<half>(half *input, half *output, int num,
                                half *minValue, half *maxValue);

}; // namespace infini

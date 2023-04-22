#include "core/common.h"
#include "cuda/cuda_common.h"
#include "utils/small_array.h"

constexpr unsigned int num_threads() { return 32 * 4; }
constexpr int thread_work_size() { return 4; }
constexpr int block_work_size() { return thread_work_size() * num_threads(); }

__global__ void _transpose_kernel(float *input, float *output, int nDims,
                                  int size, infini::SmallArray strides,
                                  infini::SmallArray outputShape) {
    int outputIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (outputIdx < size) {
        int inputIdx = 0;
        int v = outputIdx;
        for (int i = nDims - 1; i >= 0; --i) {
            inputIdx += v % outputShape.data[i] * strides.data[i];
            v /= outputShape.data[i];
        }
#if __CUDA_ARCH__ >= 350 || defined(USE_ROCM)
        output[outputIdx] = __ldg(input + inputIdx);
#else
        output[outputIdx] = input[inputIdx];
#endif
    }
}

namespace infini {
void transpose_kernel(float *input, float *output, int nDims, int size,
                      SmallArray strides, SmallArray outputShape) {
    int blocksize = block_work_size();
    int gridsize = (size + block_work_size() - 1) / block_work_size();
    _transpose_kernel<<<blocksize, gridsize>>>(input, output, nDims, size,
                                               strides, outputShape);
}

} // namespace infini

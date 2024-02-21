#include "core/common.h"
#include "cuda/cuda_common.h"
#include "cuda/cuda_utility.h"
#include "utils/small_array.h"

constexpr unsigned int num_threads() { return 32 * 4; }
constexpr int thread_work_size() { return 4; }
constexpr int block_work_size() { return thread_work_size() * num_threads(); }

template <class T>
__global__ void _transpose_kernel(void *input, void *output, int nDims,
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
        ((T *)output)[outputIdx] = ((T *)input)[inputIdx];
    }
}
#define CASE(T)                                                                \
    _transpose_kernel<DT_CUDA<T>::t>                                           \
        <<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>           \
        (input, output, nDims, size, strides, outputShape);

#define SWITCH_DTYPE(DTYPE)                                                    \
    switch (DTYPE) {                                                           \
    case 1:                                                                    \
        CASE(1)                                                                \
        break;                                                                 \
    case 2:                                                                    \
        CASE(2)                                                                \
        break;                                                                 \
    case 3:                                                                    \
        CASE(3)                                                                \
        break;                                                                 \
    case 4:                                                                    \
        CASE(4)                                                                \
        break;                                                                 \
    case 5:                                                                    \
        CASE(5)                                                                \
        break;                                                                 \
    case 6:                                                                    \
        CASE(6)                                                                \
        break;                                                                 \
    case 7:                                                                    \
        CASE(7)                                                                \
        break;                                                                 \
    case 10:                                                                   \
        CASE(10)                                                               \
        break;                                                                 \
    case 11:                                                                   \
        CASE(11)                                                               \
        break;                                                                 \
    case 12:                                                                   \
        CASE(12)                                                               \
        break;                                                                 \
    case 13:                                                                   \
        CASE(13)                                                               \
        break;                                                                 \
    case 16:                                                                   \
        CASE(16)                                                               \
        break;                                                                 \
    default:                                                                   \
        IT_TODO_HALT();                                                        \
    }

namespace infini {
void transpose_kernel(int dType, void *input, void *output, int nDims, int size,
                      SmallArray strides, SmallArray outputShape) {
    int blocksize = block_work_size();
    int gridsize = (size + block_work_size() - 1) / block_work_size();
    SWITCH_DTYPE(dType)
}

} // namespace infini

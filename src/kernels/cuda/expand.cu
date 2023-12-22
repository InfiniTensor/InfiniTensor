#include "core/common.h"
#include "cuda/cuda_common.h"
#include "cuda/cuda_utility.h"
#include "utils/small_array.h"

constexpr unsigned int num_threads() { return 32 * 4; }
constexpr int thread_work_size() { return 4; }
constexpr int block_work_size() { return thread_work_size() * num_threads(); }

template <class T>
__global__ void _expandKernel(void *input, void *output, int nDims,
                              int outputsize, infini::SmallArray inputShape,
                              infini::SmallArray outputShape) {

    int outputIdx =
        blockIdx.x * blockDim.x + threadIdx.x; // i(JKS) + j(KS) + k(S) + s
    if (outputIdx < outputsize) {
        int inputIdx = 0;  // record input index
        int temp = 1;      // stored S, KS, JKS, in order
        int tmp = 1;       // stored s,k,j,i in order
        int v = outputIdx; // v = i(JKS) + j(KS) + k(S) + s
        for (int i = nDims - 1; i >= 0; --i) {
            if (i == 0) {
                tmp = v; // i = outputIdx/(JKS)
            } else {
                tmp = v % outputShape.data[i]; // store s,k,j in order
            }
            if (inputShape.data[i] ==
                1) { // if input shape = 1, the index only equal 0
                inputIdx += 0;
            } else {
                inputIdx +=
                    tmp * temp; // otherwise +i(JKS) or j(KS) or k(S) or s
            }
            temp *= inputShape.data[i];
            v = v / outputShape.data[i];
        }
        ((T *)output)[outputIdx] = ((T *)input)[inputIdx];
    }
}

namespace infini {

#define CASE(T)                                                                \
    _expandKernel<DT_CUDA<T>::t><<<gridsize, blocksize, 0, CUDAStream::stream>>>(                     \
        input, output, nDims, outputsize, inputShape, outputShape);

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

void expandKernel(int dType, void *input, void *output, int nDims,
                  int outputsize, SmallArray inputShape,
                  SmallArray outputShape) {
    int blocksize = block_work_size();
    int gridsize = (outputsize + block_work_size() - 1) / block_work_size();
    SWITCH_DTYPE(dType)
}

} // namespace infini

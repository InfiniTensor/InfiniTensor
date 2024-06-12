#include "cuda/cuda_common.h"
#include "cuda/cuda_utility.h"
#include "utils/small_array.h"
const int repeat = 1;

template <typename T>
__global__ void
_whereKernel(void *inputX, void *inputY, const uint8_t *condition, void *output,
             int a0, int a1, int a2, int a3, int b0, int b1, int b2, int b3,
             int c0, int c1, int c2, int c3, int d0, int d1, int d2, int d3) {

    int stride1 = d2 * d3;
    int stride0 = d1 * stride1;
    int n = d0 * stride0;
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int end = (repeat * index + repeat < n ? repeat * index + repeat : n);
    for (int i = repeat * index; i < end; i++) {
        int inputXIdx = (a0 * a1 * a2 * a3 == n ? i : 0);
        int inputYIdx = (b0 * b1 * b2 * b3 == n ? i : 0);
        int conditionIdx = (c0 * c1 * c2 * c3 == n ? i : 0);

        bool aIdx = (a0 * a1 * a2 * a3 < n && a0 * a1 * a2 * a3 > 1);
        bool bIdx = (b0 * b1 * b2 * b3 < n && b0 * b1 * b2 * b3 > 1);
        bool cIdx = (c0 * c1 * c2 * c3 < n && c0 * c1 * c2 * c3 > 1);
        if (aIdx || bIdx || cIdx) {
            int d0_index = i / stride0;
            int d1_index = (i % stride0) / stride1;
            int d2_index = (i % stride1) / d3;
            int d3_index = i % d3;
            if (aIdx) {
                int a0_index = d0_index % a0;
                int a1_index = d1_index % a1;
                int a2_index = d2_index % a2;
                int a3_index = d3_index % a3;
                inputXIdx = a0_index * a1 * a2 * a3 + a1_index * a2 * a3 +
                            a2_index * a3 + a3_index;
            }
            if (bIdx) {
                int b0_index = d0_index % b0;
                int b1_index = d1_index % b1;
                int b2_index = d2_index % b2;
                int b3_index = d3_index % b3;
                inputYIdx = b0_index * b1 * b2 * b3 + b1_index * b2 * b3 +
                            b2_index * b3 + b3_index;
            }
            if (cIdx) {
                int c0_index = d0_index % c0;
                int c1_index = d1_index % c1;
                int c2_index = d2_index % c2;
                int c3_index = d3_index % c3;
                conditionIdx = c0_index * c1 * c2 * c3 + c1_index * c2 * c3 +
                               c2_index * c3 + c3_index;
            }
        }

        ((T *)output)[i] = condition[conditionIdx] ? ((T *)inputX)[inputXIdx]
                                                   : ((T *)inputY)[inputYIdx];
    }
}
#define CASE(T)                                                                \
    _whereKernel<DT_CUDA<T>::t>                                                \
        <<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(          \
            inputX, inputY, condition, output, a0, a1, a2, a3, b0, b1, b2, b3, \
            c0, c1, c2, c3, d0, d1, d2, d3);

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
__device__ int inferIndex(infini::SmallArray inputShape,
                          infini::SmallArray outputShape, int nDims, int size,
                          int outputIdx) {
    int inputIdx = 0;
    int tempInput = 1;
    int tempOutput = 1;
    for (int i = nDims - 1; i >= nDims - size; --i) {
        tempOutput = outputIdx % outputShape.data[i];
        if (inputShape.data[i] != 1) {
            inputIdx += tempInput * tempOutput;
        }
        tempInput *= inputShape.data[i];
        outputIdx /= outputShape.data[i];
    }
    return inputIdx;
}
template <typename T>
__global__ void
_whereKernel(void *inputX, void *inputY, const uint8_t *condition, void *output,
             int nDims, int outputsize, infini::SmallArray inputXShape,
             infini::SmallArray inputYShape, infini::SmallArray conditionShape,
             infini::SmallArray outputShape, int xSize, int ySize, int cSize) {

    int outputIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (outputIdx < outputsize) {
        int conditionIdx =
            inferIndex(conditionShape, outputShape, nDims, cSize, outputIdx);
        int inputXIdx =
            inferIndex(inputXShape, outputShape, nDims, xSize, outputIdx);

        int inputYIdx =
            inferIndex(inputYShape, outputShape, nDims, ySize, outputIdx);

        ((T *)output)[outputIdx] = condition[conditionIdx]
                                       ? ((T *)inputX)[inputXIdx]
                                       : ((T *)inputY)[inputYIdx];
    }
}
#define CASECurrency(T)                                                        \
    _whereKernel<DT_CUDA<T>::t>                                                \
        <<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(          \
            inputX, inputY, condition, output, nDims, outputsize, inputXShape, \
            inputYShape, conditionShape, outputShape, xSize, ySize, cSize);

#define SWITCHCurrency_DTYPE(DTYPE)                                            \
    switch (DTYPE) {                                                           \
    case 1:                                                                    \
        CASECurrency(1) break;                                                 \
    case 2:                                                                    \
        CASECurrency(2) break;                                                 \
    case 3:                                                                    \
        CASECurrency(3) break;                                                 \
    case 4:                                                                    \
        CASECurrency(4) break;                                                 \
    case 5:                                                                    \
        CASECurrency(5) break;                                                 \
    case 6:                                                                    \
        CASECurrency(6) break;                                                 \
    case 7:                                                                    \
        CASECurrency(7) break;                                                 \
    case 10:                                                                   \
        CASECurrency(10) break;                                                \
    case 11:                                                                   \
        CASECurrency(11) break;                                                \
    case 12:                                                                   \
        CASECurrency(12) break;                                                \
    case 13:                                                                   \
        CASECurrency(13) break;                                                \
    case 16:                                                                   \
        CASECurrency(16) break;                                                \
    default:                                                                   \
        IT_TODO_HALT();                                                        \
    }
namespace infini {

void whereKernel(int dTypeIndex, void *inputX, void *inputY,
                 const uint8_t *condition, void *output, int a0, int a1, int a2,
                 int a3, int b0, int b1, int b2, int b3, int c0, int c1, int c2,
                 int c3, int d0, int d1, int d2, int d3) {
    int blocksize;
    int outputsize = d0 * d1 * d2 * d3;
    if (outputsize > 511 * repeat) {
        blocksize = 1024;
    } else if (outputsize > 255 * repeat) {
        blocksize = 512;
    } else if (outputsize > 127 * repeat) {
        blocksize = 256;
    } else if (outputsize > 63 * repeat) {
        blocksize = 128;
    } else if (outputsize > 31 * repeat) {
        blocksize = 64;
    } else {
        blocksize = 32;
    }
    int gridsize = (outputsize + repeat * blocksize - 1) / (repeat * blocksize);

    SWITCH_DTYPE(dTypeIndex)
}

void whereKernel(int dTypeIndex, void *inputX, void *inputY,
                 const uint8_t *condition, void *output, int nDims,
                 int outputsize, SmallArray inputXShape, SmallArray inputYShape,
                 SmallArray conditionShape, SmallArray outputShape, int xSize,
                 int ySize, int cSize) {
    int blocksize;
    if (outputsize > 511) {
        blocksize = 1024;
    } else if (outputsize > 255) {
        blocksize = 512;
    } else if (outputsize > 127) {
        blocksize = 256;
    } else if (outputsize > 63) {
        blocksize = 128;
    } else if (outputsize > 31) {
        blocksize = 64;
    } else {
        blocksize = 32;
    }
    int gridsize = (outputsize + blocksize - 1) / blocksize;

    SWITCHCurrency_DTYPE(dTypeIndex)
}

} // namespace infini

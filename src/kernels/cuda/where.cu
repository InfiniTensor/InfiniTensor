#include "cuda/cuda_common.h"
#include "cuda/cuda_utility.h"
#include "utils/small_array.h"

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
#define CASE(T)                                                                \
    _whereKernel<DT_CUDA<T>::t>                                                \
        <<<gridsize, blocksize, 0, CUDAStream::stream>>>(                      \
            inputX, inputY, condition, output, nDims, outputsize, inputXShape, \
            inputYShape, conditionShape, outputShape, xSize, ySize, cSize);

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

    SWITCH_DTYPE(dTypeIndex)
}

} // namespace infini

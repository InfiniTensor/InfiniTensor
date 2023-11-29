#include "cuda/cuda_common.h"
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
__global__ void _whereKernel(const float *inputX, const float *inputY,
                             const uint8_t *condition, float *output, int nDims,
                             int outputsize, infini::SmallArray inputXShape,
                             infini::SmallArray inputYShape,
                             infini::SmallArray conditionShape,
                             infini::SmallArray outputShape, int xSize,
                             int ySize, int cSize) {

    int outputIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (outputIdx < outputsize) {
        int conditionIdx =
            inferIndex(conditionShape, outputShape, nDims, cSize, outputIdx);
        int inputXIdx =
            inferIndex(inputXShape, outputShape, nDims, xSize, outputIdx);

        int inputYIdx =
            inferIndex(inputYShape, outputShape, nDims, ySize, outputIdx);

        output[outputIdx] =
            condition[conditionIdx] ? inputX[inputXIdx] : inputY[inputYIdx];
    }
}

namespace infini {
void whereKernel(const float *inputX, const float *inputY,
                 const uint8_t *condition, float *output, int nDims,
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
    _whereKernel<<<gridsize, blocksize>>>(
        inputX, inputY, condition, output, nDims, outputsize, inputXShape,
        inputYShape, conditionShape, outputShape, xSize, ySize, cSize);
}
} // namespace infini

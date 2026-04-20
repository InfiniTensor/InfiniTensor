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
template <typename T>
__global__ void
_whereKernel(const T *inputX, const T *inputY, const uint8_t *condition,
             T *output, int nDims, int outputsize,
             infini::SmallArray inputXShape, infini::SmallArray inputYShape,
             infini::SmallArray conditionShape, infini::SmallArray outputShape,
             int xSize, int ySize, int cSize) {

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
template <typename T>
__global__ void _where_const_Kernel(const T *inputX, const T *inputY,
                                    const uint8_t *condition, T *output,
                                    int outputsize, bool const_x, bool const_y,
                                    bool const_c) {

    int outputIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (outputIdx >= outputsize)
        return;

    // 根据标志位决定索引
    int inputXIdx = const_x ? 0 : outputIdx;
    int inputYIdx = const_y ? 0 : outputIdx;
    int conditionIdx = const_c ? 0 : outputIdx;

    // where逻辑
    output[outputIdx] =
        condition[conditionIdx] ? inputX[inputXIdx] : inputY[inputYIdx];
}
namespace infini {
template <typename Tdata>
void whereKernel(const Tdata *inputX, const Tdata *inputY,
                 const uint8_t *condition, Tdata *output, int nDims,
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
    bool const_x = true;
    bool const_y = true;
    bool const_c = true;

    for (int i = 0; i < nDims; i++) {
        if (inputYShape.data[i] != conditionShape.data[i]) {
            const_x = false;
            break;
        }
    }
    for (int i = 0; i < nDims; i++) {
        if (inputXShape.data[i] != conditionShape.data[i]) {
            const_y = false;
            break;
        }
    }
    for (int i = 0; i < nDims; i++) {
        if (inputYShape.data[i] != inputXShape.data[i]) {
            const_c = false;
            break;
        }
    }
    if (xSize == 0 && const_x) {

        _where_const_Kernel<Tdata>
            <<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(
                inputX, inputY, condition, output, outputsize, true, false,
                false);
    } else if (ySize == 0 && const_y) {

        _where_const_Kernel<Tdata>
            <<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(
                inputX, inputY, condition, output, outputsize, false, true,
                false);
    } else if (cSize == 0 && const_c) {

        _where_const_Kernel<Tdata>
            <<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(
                inputX, inputY, condition, output, outputsize, false, false,
                true);
    } else {

        _whereKernel<Tdata>
            <<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(
                inputX, inputY, condition, output, nDims, outputsize,
                inputXShape, inputYShape, conditionShape, outputShape, xSize,
                ySize, cSize);
    }
}

template void whereKernel<float>(const float *inputX, const float *inputY,
                                 const uint8_t *condition, float *output,
                                 int nDims, int outputsize,
                                 SmallArray inputXShape, SmallArray inputYShape,
                                 SmallArray conditionShape,
                                 SmallArray outputShape, int xSize, int ySize,
                                 int cSize);
template void whereKernel<half>(const half *inputX, const half *inputY,
                                const uint8_t *condition, half *output,
                                int nDims, int outputsize,
                                SmallArray inputXShape, SmallArray inputYShape,
                                SmallArray conditionShape,
                                SmallArray outputShape, int xSize, int ySize,
                                int cSize);

} // namespace infini

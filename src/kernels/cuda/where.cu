#include "cuda/cuda_common.h"
#include "utils/small_array.h"

__global__ void _where_kernel(const float *inputX, const float *inputY,
                              const uint8_t *condition, float *output,
                              int nDims, int outputsize,
                              infini::SmallArray inputXShape,
                              infini::SmallArray inputYShape,
                              infini::SmallArray conditionShape,
                              infini::SmallArray outputShape) {

    int outputIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (outputIdx < outputsize) {
        int inputXIdx = 0;
        int temp_inputX = 1;

        int inputYIdx = 0;
        int temp_inputY = 1;

        int conditionIdx = 0;
        int temp_condition = 1;

        int tmp = 1;       // stored s,k,j,i in order
        int v = outputIdx; // v = i(JKS) + j(KS) + k(S) + s
        for (int i = nDims - 1; i >= 0; --i) {
            if (i == 0) {
                tmp = v; // i = outputIdx/(JKS)
            } else {
                tmp = v % outputShape.data[i]; // store s,k,j in order
            }
            if (inputXShape.data[i] == 1) {
                inputXIdx += 0;
            } else {
                inputXIdx +=
                    tmp *
                    temp_inputX; // otherwise +i(JKS) or j(KS) or k(S) or s
            }
            temp_inputX *= inputXShape.data[i];
            //----------------------------
            if (inputYShape.data[i] == 1) {
                inputYIdx += 0;
            } else {
                inputYIdx +=
                    tmp *
                    temp_inputY; // otherwise +i(JKS) or j(KS) or k(S) or s
            }
            temp_inputY *= inputYShape.data[i];
            //--------------------------
            if (conditionShape.data[i] == 1) {
                conditionIdx += 0;
            } else {
                conditionIdx +=
                    tmp *
                    temp_condition; // otherwise +i(JKS) or j(KS) or k(S) or s
            }
            temp_condition *= conditionShape.data[i];
            //-------------------------
            v = v / outputShape.data[i];
        }
        output[outputIdx] =
            condition[conditionIdx] ? inputX[inputXIdx] : inputY[inputYIdx];
    }
}

namespace infini {
void where_kernel(const float *inputX, const float *inputY,
                  const uint8_t *condition, float *output, int nDims,
                  infini::SmallArray inputXShape,
                  infini::SmallArray inputYShape,
                  infini::SmallArray conditionShape,
                  infini::SmallArray outputShape) {
    int outputsize = 1;

    for (int i = 0; i < nDims; i++) {
        outputsize *= outputShape.data[i];
    }
    int blocksize = 32 * 16;
    int gridsize = (outputsize + blocksize - 1) / blocksize;
    _where_kernel<<<gridsize, blocksize>>>(
        inputX, inputY, condition, output, nDims, outputsize, inputXShape,
        inputYShape, conditionShape, outputShape);
}
} // namespace infini
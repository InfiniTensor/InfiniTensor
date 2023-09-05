#include "cuda/cuda_common.h"
#include "utils/small_array.h"

__global__ void _where_kernel(const float *inputx, const float *inputy,
                              const int *condition, float *output, int nDims,
                              int outputsize, infini::SmallArray inputxShape,
                              infini::SmallArray inputyShape,
                              infini::SmallArray conditionShape,
                              infini::SmallArray outputShape) {

    int outputIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (outputIdx < outputsize) {
        int inputxIdx = 0;
        int temp_inputx = 1;

        int inputyIdx = 0;
        int temp_inputy = 1;

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
            if (inputxShape.data[i] == 1) {
                inputxIdx += 0;
            } else {
                inputxIdx +=
                    tmp *
                    temp_inputx; // otherwise +i(JKS) or j(KS) or k(S) or s
            }
            temp_inputx *= inputxShape.data[i];
            //----------------------------
            if (inputyShape.data[i] == 1) {
                inputyIdx += 0;
            } else {
                inputyIdx +=
                    tmp *
                    temp_inputy; // otherwise +i(JKS) or j(KS) or k(S) or s
            }
            temp_inputy *= inputyShape.data[i];
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
            condition[conditionIdx] ? inputx[inputxIdx] : inputy[inputyIdx];
    }
}

namespace infini {
void where_kernel(const float *inputx, const float *inputy,
                  const int *condition, float *output, int nDims,
                  infini::SmallArray inputxShape,
                  infini::SmallArray inputyShape,
                  infini::SmallArray conditionShape,
                  infini::SmallArray outputShape) {
    int outputsize = 1;

    for (int i = 0; i < nDims; i++) {
        outputsize *= outputShape.data[i];
    }
    int blocksize = 32 * 16;
    int gridsize = (outputsize + blocksize - 1) / blocksize;
    _where_kernel<<<gridsize, blocksize>>>(
        inputx, inputy, condition, output, nDims, outputsize, inputxShape,
        inputyShape, conditionShape, outputShape);
}
} // namespace infini

#include "core/common.h"
#include "cuda/cuda_common.h"
#include "utils/small_array.h"

constexpr unsigned int num_threads() { return 32 * 4; }
constexpr int thread_work_size() { return 4; }
constexpr int block_work_size() { return thread_work_size() * num_threads(); }

__global__ void _expand_kernel(float *input, float *output, int nDims,
                               int outputsize, infini::SmallArray inputShape,
                               infini::SmallArray outputShape) {
    // inputShape 存储输入形状，outputShape存储目标输出形状 ，nDims存储输入维度
    int outputIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (outputIdx < outputsize) {
        int inputIdx = 0;  //记录输入的index
        int temp = 1;      //存储依次是S,KS,JKS
        int tmp = 1;       //存储依次是s,k,j,i
        int v = outputIdx; // v = i(JKS) + j(KS) + k(S) + s
        for (int i = nDims - 1; i >= 0; --i) {
            if (i == 0) {
                tmp = v; //存储最后一次是i
            } else {
                tmp = v % outputShape.data[i]; //存储是s,k,j
            }
            if (inputShape.data[i] ==
                1) { //如果输入维度=1，说明变量的可选范围只能是0
                inputIdx += 0;
            } else {
                inputIdx +=
                    tmp * temp; //否则+i(JKS) 或者 j(KS) 或者 k(S) 或者 s
            }
            temp *= inputShape.data[i];
            v = v / outputShape.data[i];
        }
        output[outputIdx] = input[inputIdx];
    }
}

namespace infini {
void expand_kernel(float *input, float *output, int nDims, int outputsize,
                   SmallArray inputShape, SmallArray outputShape) {
    int blocksize = block_work_size();
    int gridsize = (outputsize + block_work_size() - 1) / block_work_size();
    _expand_kernel<<<gridsize, blocksize>>>(input, output, nDims, outputsize,
                                            inputShape, outputShape);
}

} // namespace infini

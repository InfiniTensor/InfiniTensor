#include "core/common.h"
#include "cuda/cuda_common.h"
#include "utils/small_array.h"

constexpr unsigned int num_threads() { return 32 * 4; }
constexpr int thread_work_size() { return 4; }
constexpr int block_work_size() { return thread_work_size() * num_threads(); }

__global__ void _expand_kernel(float *input, float *output, int nDims,
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

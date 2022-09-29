#include "cuda/cuda_common.h"

__global__ void _extend_kernel(float *in, float *out, int blockSize,
                               int blockSizeOuter, int oSize) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= oSize)
        return;

    int stride = blockDim.x * gridDim.x;
    while (index < oSize) {
        auto iIdx = index % blockSize + index / blockSizeOuter * blockSize;
        out[index] = in[iIdx];
        index += stride;
    }
}

namespace infini {
void extend_kernel(float *in, float *out, int blockSize, int blockSizeOuter,
                   int oSize) {
    int blocksize = 32 * 16;
    int gridsize = (oSize + blocksize - 1) / blocksize;
    _extend_kernel<<<blocksize, gridsize>>>(in, out, blockSize, blockSizeOuter,
                                            oSize);
}
} // namespace infini

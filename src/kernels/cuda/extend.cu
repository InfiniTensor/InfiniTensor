#include "cuda/cuda_common.h"

template <typename T>
__global__ void _extend_kernel(T *in, T *out, int blockSize,
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
template <typename T>
void extend_impl(T *in, T *out, int blockSize, int blockSizeOuter,
                   int oSize) {
    int blocksize = 32 * 16;
    int gridsize = (oSize + blocksize - 1) / blocksize;
    _extend_kernel<T>
        <<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(
        in, out, blockSize, blockSizeOuter, oSize);
}
void extend_kernel(float *in, float *out, int blockSize, int blockSizeOuter,
                   int oSize) {
    extend_impl(in, out, blockSize, blockSizeOuter, oSize);
}
void extend_kernel(half *in, half *out, int blockSize, int blockSizeOuter,
                   int oSize) {
    extend_impl(in, out, blockSize, blockSizeOuter, oSize);
}
} // namespace infini

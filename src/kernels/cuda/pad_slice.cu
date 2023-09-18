#include "core/data_type.h"
#include "cuda/cuda_common.h"
#include "cuda/cuda_pad_slice.h"

__device__ int WholeTensorOffset2PartTensorOffset(int wholeOffset,
                                                  TransMetaData metaData,
                                                  int nDims) {
    int offset = 0;
    for (int i = nDims - 1; i >= 0; --i) {
        auto wholePos = wholeOffset % metaData.wholeNDim[i];
        auto pos = wholePos - metaData.begNum[i];
        // if pos belongs to pad range, then return -1
        if (pos < 0 || pos >= metaData.partNDim[i])
            return -1;
        wholeOffset = wholeOffset / metaData.wholeNDim[i];

        offset += pos * metaData.partStride[i];
    }

    return offset;
}

template <typename T>
__global__ void _pad_slice_kernel(T *part, T *whole, TransMetaData metaData,
                                  int nDims, int num, bool isPad) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num)
        return;

    int stride = blockDim.x * gridDim.x;
    while (tid < num) {
        int offset = WholeTensorOffset2PartTensorOffset(tid, metaData, nDims);
        if (isPad)
            if (offset < 0)
                whole[tid] = 0;
            else
                whole[tid] = part[offset];
        else
            part[offset] = whole[tid];
        tid += stride;
    }
}

namespace infini {
void pad_slice_kernel(void *partData, void *wholeData,
                      const TransMetaData &metadata, int nDims, int num,
                      bool isPad) {
    int blockSize = 32 * 16;
    int gridSize = (num + blockSize - 1) / blockSize;
    if (metadata.DType == DataType::Int64.getIndex()) {
        _pad_slice_kernel<int64_t>
            <<<gridSize, blockSize>>>((int64_t *)partData, (int64_t *)wholeData,
                                      metadata, nDims, num, isPad);
    } else if (metadata.DType == DataType::Float32.getIndex()) {
        _pad_slice_kernel<float><<<gridSize, blockSize>>>(
            (float *)partData, (float *)wholeData, metadata, nDims, num, isPad);
    }
}
} // namespace infini

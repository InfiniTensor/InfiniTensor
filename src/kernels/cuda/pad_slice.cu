#include "core/data_type.h"
#include "cuda/cuda_common.h"
#include "cuda/cuda_pad_slice.h"
#include "cuda/cuda_utility.h"

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
__global__ void _pad_slice_kernel(void *part, void *whole,
                                  TransMetaData metaData, int nDims, int num,
                                  bool isPad) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num) {
        return;
    }

    int stride = blockDim.x * gridDim.x;
    while (tid < num) {
        int offset = WholeTensorOffset2PartTensorOffset(tid, metaData, nDims);
        if (isPad) {
            if (offset < 0) {
                ((T *)whole)[tid] = 0;
            } else {
                ((T *)whole)[tid] = ((T *)part)[offset];
            }
        } else if (offset >= 0) {
            ((T *)part)[offset] = ((T *)whole)[tid];
        }
        tid += stride;
    }
}

namespace infini {
#define CASE(T)                                                                \
    _pad_slice_kernel<DT_CUDA<T>::t><<<gridSize, blockSize>>>(                 \
        partData, wholeData, metadata, nDims, num, isPad);

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

void pad_slice_kernel(void *partData, void *wholeData,
                      const TransMetaData &metadata, int nDims, int num,
                      bool isPad) {
    int blockSize = 32 * 16;
    int gridSize = (num + blockSize - 1) / blockSize;
    int dType = metadata.DType;
    SWITCH_DTYPE(dType)
}
} // namespace infini

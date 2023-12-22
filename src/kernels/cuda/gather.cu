#include "cuda/cuda_common.h"
#include "cuda/gather.h"

template <typename T>
__device__ T gatheredOffset2Offset(int gOffset,
                                   infini::GatherMetaData metaData) {
    T offset = 0;
    for (int i = metaData.inNDim - 1, k = metaData.outNDim - 1; i >= 0; --i) {
        T idx = 0;
        if (i == metaData.axis) {
            T idxOffset = 0;
            for (int j = metaData.idxNDim - 1; j >= 0; --j) {
                T p = gOffset % metaData.idxDim[j];
                gOffset = gOffset / metaData.idxDim[j];
                idxOffset += p * metaData.idxStride[j];
            }

            idx = static_cast<T *>(metaData.indexValue)[idxOffset];
            k = k - metaData.idxNDim;

        } else {
            idx = gOffset % metaData.outDim[k];
            gOffset = gOffset / metaData.outDim[k];
            --k;
        }
        offset += idx * metaData.inStride[i];
    }
    return offset;
}

template <typename dataT, typename T>
__global__ void _gather_kernel(dataT *in, dataT *out,
                               infini::GatherMetaData metaData, size_t num) {
    T tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < num) {
        T offset = gatheredOffset2Offset<T>(tid, metaData);
        out[tid] = in[offset];
    }
}

namespace infini {
template <typename T>
void gather_kernel(T *in, T *out, GatherMetaData metaData, size_t num) {
    int blockSize = 32 * 16;
    int gridSize = (num + blockSize - 1) / blockSize;
    if (metaData.indexType == DataType::Int64) {
        _gather_kernel<T, int64_t>
            <<<gridSize, blockSize, 0, CUDAStream::stream>>>(in, out, metaData, num);
    } else {
        _gather_kernel<T, int><<<gridSize, blockSize, 0, CUDAStream::stream>>>(in, out, metaData, num);
    }
}
template void gather_kernel<float>(float *in, float *out,
                                   GatherMetaData metaData, size_t num);
template void gather_kernel<half>(half *in, half *out, GatherMetaData metaData,
                                  size_t num);
template void gather_kernel<int8_t>(int8_t *in, int8_t *out,
                                    GatherMetaData metaData, size_t num);
template void gather_kernel<uint8_t>(uint8_t *in, uint8_t *out,
                                     GatherMetaData metaData, size_t num);
} // namespace infini

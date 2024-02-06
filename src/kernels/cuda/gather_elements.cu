#include "cuda/cuda_common.h"
#include "cuda/gather.h"

template <typename Tind>
__device__ Tind tid2Offset(Tind tid, infini::GatherMetaData metaData) {
    Tind offset = 0;
    Tind gOffset = tid;
    for (int i = metaData.inNDim - 1; i >= 0; --i) {
        if (i == metaData.axis) {
            Tind idx = static_cast<Tind *>(metaData.indexValue)[tid];
            offset += idx * metaData.inStride[i];
        } else {
            Tind p = gOffset % metaData.idxDim[i];
            offset += p * metaData.inStride[i];
        }

        gOffset = gOffset / metaData.idxDim[i];
    }

    return offset;
}

template <typename T, typename Tind>
__global__ void _gather_elements_kernel(T *in, T *out,
                                        infini::GatherMetaData metaData,
                                        size_t num) {
    Tind tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (tid < num) {
        Tind offset = tid2Offset<Tind>(tid, metaData);
        out[tid] = in[offset];
        tid += stride;
    }
}

namespace infini {
void gather_elements_kernel(void *in, void *out, GatherMetaData metaData,
                            size_t num) {
    int blockSize = 1024;
    int gridSize = (num + blockSize - 1) / blockSize;
    if (metaData.dataType == DataType::Float32 &&
        metaData.indexType == DataType::Int64) {
        _gather_elements_kernel<float, int64_t>
            <<<gridSize, blockSize, 0, CUDAStream::getCurrentStream()>>>(
            reinterpret_cast<float *>(in), reinterpret_cast<float *>(out),
            metaData, num);
    } else if (metaData.dataType == DataType::Int32 &&
               metaData.indexType == DataType::Int64) {
        _gather_elements_kernel<int, int64_t>
            <<<gridSize, blockSize, 0, CUDAStream::getCurrentStream()>>>(
            reinterpret_cast<int *>(in), reinterpret_cast<int *>(out), metaData,
            num);
    } else if (metaData.dataType == DataType::Float32 &&
               metaData.indexType == DataType::Int32) {
        _gather_elements_kernel<float, int>
            <<<gridSize, blockSize, 0, CUDAStream::getCurrentStream()>>>(
            reinterpret_cast<float *>(in), reinterpret_cast<float *>(out),
            metaData, num);
    } else if (metaData.dataType == DataType::Int32 &&
               metaData.indexType == DataType::Int32) {
        _gather_elements_kernel<int, int>
            <<<gridSize, blockSize, 0, CUDAStream::getCurrentStream()>>>(
            reinterpret_cast<int *>(in), reinterpret_cast<int *>(out), metaData,
            num);
    } else {
        IT_TODO_HALT_MSG(
            "GatherElements Cuda Kernel: Unsupported data type.\n");
    }
}
} // namespace infini

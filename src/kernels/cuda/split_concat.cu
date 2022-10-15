#include "cuda/cuda_common.h"
#include "cuda/cuda_split_concat.h"

int getMultiProcessorCount() {
    int cur_device;
    checkCudaError(cudaGetDevice(&cur_device));

    struct cudaDeviceProp prop;
    checkCudaError(cudaGetDeviceProperties(&prop, cur_device));
    return prop.multiProcessorCount;
}

__host__ __device__ int
elementIdx2ComposedIdx(int elementIndex, int dimBgNo, int dimSize, int dim,
                       int nDim, ComposedTensorMetadata wholeMeta) {
    int offset = 0;

#pragma unroll
    for (int i = nDim - 1; i >= 1; --i) {
        int size = (i == dim) ? dimSize : wholeMeta.dimSize[i];
        int p = elementIndex % size;
        int oP = (i == dim) ? (p + dimBgNo) : p;
        elementIndex = (elementIndex - p) / size;
        offset += oP * wholeMeta.stride[i];
    }

    return offset + elementIndex * wholeMeta.stride[0];
}

__global__ void _split_concat_kernel(ElementTensorMetadata elemMeta,
                                     ComposedTensorMetadata compMeta, int dim,
                                     int nDims, bool isSplit) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nElements = elemMeta.nElements[blockIdx.y];
    if (tid >= nElements)
        return;

    auto dimBgNo = elemMeta.dimBgNo[blockIdx.y];
    auto dimSize = elemMeta.dimSize[blockIdx.y];
    float *elemData = elemMeta.data[blockIdx.y];
    int stride = gridDim.x * blockDim.x;

    while (tid < nElements) {
        int Offset =
            elementIdx2ComposedIdx(tid, dimBgNo, dimSize, dim, nDims, compMeta);
        // copy data from input to output
        // for split:input is composed tensor;for concat:input is element
        // tensors.
        if (isSplit)
            elemData[tid] = compMeta.data[Offset];
        else
            compMeta.data[Offset] = elemData[tid];
        tid += stride;
    }
}

namespace infini {

void split_concat_kernel(const ElementTensorMetadata &eleMeta,
                         const ComposedTensorMetadata &compMeta, int dim,
                         int batchSize, int nDims, bool isSplit) {
    dim3 blockSize = dim3(32 * 16);

    //  y dim is number of tensors.
    dim3 gridSize(getMultiProcessorCount(), batchSize);

    _split_concat_kernel<<<gridSize, blockSize>>>(eleMeta, compMeta, dim, nDims,
                                                  isSplit);
}

} // namespace infini

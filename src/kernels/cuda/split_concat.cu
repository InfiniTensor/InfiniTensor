#include "cuda/cuda_common.h"
#include "cuda/cuda_split_concat.h"
template <typename T>
__host__ __device__ int
elementIdx2ComposedIdx(int elementIndex, int dimBgNo, int dimSize, int dim,
                       int nDim, ComposedTensorMetadata<T> wholeMeta) {
    int offset = 0;

    // COMP(x0,...,xk,...,xn-1) = ELMT[xk / d](x0,...,xk % d,...xn-1)
    // where k=dim, n=ndim, d=dimSize is the splited length of
    // dimension dim
#pragma unroll
    // Interate through n-1 to 1
    for (int i = nDim - 1; i >= 1; --i) {
        int size = (i == dim) ? dimSize : wholeMeta.dimSize[i];
        int p = elementIndex % size;
        // dimBgNo move the pointer to correct location in composed data
        // corresponding to current element, with repect to the splitted
        // dimension dim
        int oP = (i == dim) ? (p + dimBgNo) : p;
        elementIndex = (elementIndex - p) / size;
        offset += oP * wholeMeta.stride[i];
    }
    // Deal with i = 0
    int oP = (dim == 0) ? (elementIndex + dimBgNo) : elementIndex;
    return offset + oP * wholeMeta.stride[0];
}
template <typename T>
__global__ void _split_concat_kernel(ElementTensorMetadata<T> elemMeta,
                                     ComposedTensorMetadata<T> compMeta,
                                     int dim, int nDims, bool isSplit) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nElements = elemMeta.nElements[blockIdx.y];
    if (tid >= nElements)
        return;

    auto dimBgNo = elemMeta.dimBgNo[blockIdx.y];
    auto dimSize = elemMeta.dimSize[blockIdx.y];
    T *elemData = elemMeta.data[blockIdx.y];

    int Offset =
        elementIdx2ComposedIdx<T>(tid, dimBgNo, dimSize, dim, nDims, compMeta);
    // copy data from input to output
    // for split:input is composed tensor;for concat:input is element
    // tensors.
    if (isSplit)
        elemData[tid] = compMeta.data[Offset];
    else
        compMeta.data[Offset] = elemData[tid];
}

namespace infini {

// TODO: when dim=0, the operation can be executed in-place
void split_concat_kernel(const ElementTensorMetadata<float> &eleMeta,
                         const ComposedTensorMetadata<float> &compMeta, int dim,
                         int batchSize, int nDims, bool isSplit) {
    dim3 blockSize = dim3(32 * 16);
    // gridsize = max_n_elements / blockSize
    int max_n_elements =
        *std::max_element(eleMeta.nElements, eleMeta.nElements + batchSize);
    int gridDimX = (max_n_elements - 1) / (32 * 16) + 1;
    // each y is a split among the batch
    dim3 gridSize(gridDimX, batchSize);

    _split_concat_kernel<<<gridSize, blockSize, 0, CUDAStream::stream>>>(eleMeta, compMeta, dim, nDims,
                                                  isSplit);
}
void split_concat_kernel(const ElementTensorMetadata<half> &eleMeta,
                         const ComposedTensorMetadata<half> &compMeta, int dim,
                         int batchSize, int nDims, bool isSplit) {
    dim3 blockSize = dim3(32 * 16);
    // gridsize = max_n_elements / blockSize
    int max_n_elements =
        *std::max_element(eleMeta.nElements, eleMeta.nElements + batchSize);
    int gridDimX = (max_n_elements + 32 * 16 - 1) / (32 * 16);
    // each y is a split among the batch
    dim3 gridSize(gridDimX, batchSize);

    _split_concat_kernel<<<gridSize, blockSize, 0, CUDAStream::stream>>>(eleMeta, compMeta, dim, nDims,
                                                  isSplit);
}

} // namespace infini

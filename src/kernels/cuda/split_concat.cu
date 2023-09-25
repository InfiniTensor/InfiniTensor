#include "cuda/cuda_common.h"
#include "cuda/cuda_split_concat.h"

__host__ __device__ int
elementIdx2ComposedIdx(int elementIndex, int dimBgNo, int dimSize, int dim,
                       int nDim, ComposedTensorMetadata wholeMeta) {
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

    int Offset =
        elementIdx2ComposedIdx(tid, dimBgNo, dimSize, dim, nDims, compMeta);
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
void split_concat_kernel(const ElementTensorMetadata &eleMeta,
                         const ComposedTensorMetadata &compMeta, int dim,
                         int batchSize, int nDims, bool isSplit) {
    dim3 blockSize = dim3(32 * 16);
    // gridsize =n_elements / blockSize
    int gridDimX = (eleMeta.nElements[0] - 1) / (32 * 16) + 1;
    // each y is a split among the batch
    dim3 gridSize(gridDimX, batchSize);

    _split_concat_kernel<<<gridSize, blockSize>>>(eleMeta, compMeta, dim, nDims,
                                                  isSplit);
}

} // namespace infini

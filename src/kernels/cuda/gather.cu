#include "cuda/cuda_common.h"
#include "cuda/gather.h"

__device__ int gatheredOffset2Offset(int gOffset, GatherMetaData metaData) {
    int offset = 0;
    for (int i = metaData.inNDim - 1, k = metaData.outNDim - 1; i >= 0; --i) {
        int idx = 0;
        if (i == metaData.axis) {
            int idxOffset = 0;
            for (int j = metaData.idxNDim - 1; j >= 0; --j) {
                int p = gOffset % metaData.idxDim[j];
                gOffset = gOffset / metaData.idxDim[j];
                idxOffset += p * metaData.idxStride[j];
            }

            idx = metaData.indexValue[idxOffset];
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

__global__ void _gather_kernel(float *in, float *out, GatherMetaData metaData,
                               int num) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (tid < num) {
        int offset = gatheredOffset2Offset(tid, metaData);
        out[tid] = in[offset];
        tid += stride;
    }
}

namespace infini {
void gather_kernel(float *in, float *out, GatherMetaData metaData, int num) {
    int blockSize = 32 * 16;
    int gridSize = (num + blockSize - 1) / blockSize;

    _gather_kernel<<<gridSize, blockSize>>>(in, out, metaData, num);
}
} // namespace infini

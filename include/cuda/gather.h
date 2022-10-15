#pragma once

typedef struct {
    int *indexValue;
    int axis;
    int inNDim;
    int outNDim;
    int idxNDim;
    int outDim[4];
    int idxDim[4];
    int idxStride[4];
    int inStride[4];
} GatherMetaData;

namespace infini {
void gather_kernel(float *in, float *out, GatherMetaData metaData, int num);
}

#pragma once
#include "core/data_type.h"

namespace infini {
struct GatherMetaData {
    void *indexValue;
    DataType indexType;
    int axis;
    int inNDim;
    int outNDim;
    int idxNDim;
    int outDim[4];
    int idxDim[4];
    int idxStride[4];
    int inStride[4];
};

void gather_kernel(float *in, float *out, GatherMetaData metaData, size_t num);
} // namespace infini

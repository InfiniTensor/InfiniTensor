#pragma once
#include "cuda/cuda_common.h"

typedef struct {
    int nDims;
    int oDims[4];
    int inDims[4];
    int inStride[4];
    float scale[4];
    float roiS[4];
    float roiE[4];
} MetaData;

namespace infini {
void resize_kernel_nearest(float *in, float *out, const MetaData &metaData,
                           size_t num, int coordinateMode, int nearestMode);
void resize_kernel_linear(float *in, float *out, const MetaData &metaData,
                          size_t num, int coordinateMode);
void resize_kernel_cubic(float *in, float *out, const MetaData &metaData,
                         size_t num, int coordinateMode);
} // namespace infini

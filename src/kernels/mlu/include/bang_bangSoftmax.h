#pragma once
#include "cnnl.h"
namespace infini {
void softmaxKernel(cnnlHandle_t handle, float *mlu_destination, float *mlu_src,
                   int nDim, int axis, int othersize, int frontsize,
                   int dimsize, int stride);

}; // namespace infini

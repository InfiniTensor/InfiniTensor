#pragma once
#include "cnnl.h"
namespace infini {
void softmaxKernel(cnnlHandle_t handle, float *mlu_destination, float *mlu_src, int othersize, int dimsize, int frontsize, int stride, int axis, int nDim);

}; // namespace infini

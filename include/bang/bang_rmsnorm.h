#pragma once
#include "cnnl.h"
namespace infini {
void rmsNormKernel(cnnlHandle_t handle, float *mlu_destination, float *mlu_src,
                   float *mlu_weight, int othersize, int dimsize, float eps);
}; // namespace infini

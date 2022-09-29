#pragma once
#include "cnnl.h"
namespace infini {
void div_kernel(cnnlHandle_t handle, const float *input1, const float *input2,
                float *output, const uint32_t num);

}; // namespace infini

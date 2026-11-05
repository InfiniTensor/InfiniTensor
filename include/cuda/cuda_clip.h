#pragma once

#include "operators/unary.h"
#include <cuda_fp16.h>

namespace infini {
void clip_kernel(float *input, float *output, int num, float minValue,
                 float maxValue);
void clip_kernel(half *input, half *output, int num, float minValue,
                 float maxValue);

}; // namespace infini

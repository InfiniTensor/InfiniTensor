#pragma once
#include "operators/unary.h"

namespace infini {
void InstanceNormKernel(const float *input, const float *scale,
                        const float *bias, float *output, int N, int C,
                        int inner_size, float eps);
void InstanceNormKernel(const half *input, const half *scale, const half *bias,
                        half *output, int N, int C, int inner_size, float eps);
}; // namespace infini

#pragma once
#include "operators/unary.h"

namespace infini {
void LaynormKernel(const float *input, const float *scale, const float *bias,
                   const float eps, int size, int scaleSize, int biasSize,
                   const int dimsize, const int stride, float *output);

}; // namespace infini

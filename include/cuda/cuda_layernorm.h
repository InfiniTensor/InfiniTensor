#pragma once
#include "operators/unary.h"

namespace infini {
void LaynormKernel(const float *input, const float *scale, const float eps,
                   int size, int scaleSize, const int dimsize, const int stride,
                   float *output, const float *bias, int biasSize);
void LaynormKernel(const float *input, const float *scale, const float eps,
                   int size, int scaleSize, const int dimsize, const int stride,
                   float *output);
void LaynormKernel(const half *input, const half *scale, const half eps,
                   int size, int scaleSize, const int dimsize, const int stride,
                   half *output, const half *bias, int biasSize);
void LaynormKernel(const half *input, const half *scale, const half eps,
                   int size, int scaleSize, const int dimsize, const int stride,
                   half *output);
}; // namespace infini

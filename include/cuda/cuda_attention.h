#pragma once
#include "operators/unary.h"

namespace infini {
void attentionKernel(const float *inputQ, const float *inputK, const float *inputV, int N, int d, float *output);

}; // namespace infini

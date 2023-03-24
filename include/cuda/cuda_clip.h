#pragma once

#include "operators/unary.h"

namespace infini {
void clip_kernel(float *input, float *output, int num, float minValue,
                 float maxValue);

}; // namespace infini

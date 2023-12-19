#pragma once
#include "operators/dynamic_quantize_linear.h"

namespace infini {
void dynamicQuantizeLinearKernel(float *input, uint8_t *outputY, float *yScale,
                                 uint8_t *yZeroPoint, int size);
}; // namespace infini

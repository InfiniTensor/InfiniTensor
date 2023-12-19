#pragma once
#include "operators/dequantize_linear.h"

namespace infini {
void DequantizeLinearKernel(const uint8_t *inputX, const float *inputScale,
                            float *output, const int dimsize, const int stride,
                            const uint8_t *inputZeroPoint, const int size);
void DequantizeLinearKernel(const uint8_t *inputX, const float *inputScale,
                            float *output, const int dimsize, const int stride,
                            const int size);
void DequantizeLinearKernel(const uint8_t *inputX, const half *inputScale,
                            half *output, const int dimsize, const int stride,
                            const uint8_t *inputZeroPoint, const int size);
void DequantizeLinearKernel(const uint8_t *inputX, const half *inputScale,
                            half *output, const int dimsize, const int stride,
                            const int size);
}; // namespace infini

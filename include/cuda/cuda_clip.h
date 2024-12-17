#pragma once

#include "operators/unary.h"

namespace infini {
template <typename T>
void clip_kernel(T *input, T *output, int num, T *minValue, T *maxValue);

}; // namespace infini

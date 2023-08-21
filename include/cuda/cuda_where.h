#pragma once

#include "operators/unary.h"

namespace infini {
void where_kernel(const float *input, const float *other,
                  const float *condition, float *output, int size);

}; // namespace infini

#pragma once
#include "operators/unary.h"
#include "utils/small_array.h"

namespace infini {

void range_kernel(const float start, const float limit, const float delta,
                  float *output, int size);
}; // namespace infini

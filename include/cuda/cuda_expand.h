#pragma once

#include "operators/unary.h"
#include "utils/small_array.h"
namespace infini {
void expand_kernel(float *input, float *output, int nDims, int outputsize,
                   SmallArray inputShape, SmallArray outputShape);

}; // namespace infini
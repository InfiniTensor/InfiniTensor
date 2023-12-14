#pragma once

#include "operators/unary.h"
#include "utils/small_array.h"
namespace infini {
void expandKernel(int dType, void *input, void *output, int nDims,
                  int outputsize, SmallArray inputShape,
                  SmallArray outputShape);

}; // namespace infini

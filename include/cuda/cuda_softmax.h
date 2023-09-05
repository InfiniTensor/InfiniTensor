#pragma once
#include "utils/small_array.h"
namespace infini {
void softmax_kernel(float *input, float *output, int size,
                    SmallArray inputShape, int axis, int nDims);
}

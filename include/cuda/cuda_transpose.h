#pragma once

#include "operators/transpose.h"
#include "utils/small_array.h"

namespace infini {

void transpose_kernel(float *input, float *output, int nDims, int size,
                      SmallArray strides, SmallArray outputShape);

}; // namespace infini
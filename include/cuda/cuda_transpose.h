#pragma once

#include "operators/transpose.h"
#include "utils/small_array.h"

namespace infini {

void transpose_kernel(int dType, void *input, void *output, int nDims, int size,
                      SmallArray strides, SmallArray outputShape);
void transpose_nchw2nhcw(void *input, void *output, int N, int C, int H, int W);
}; // namespace infini

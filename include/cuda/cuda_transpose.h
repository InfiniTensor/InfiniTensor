#pragma once

#include "operators/transpose.h"
#include "utils/small_array.h"

namespace infini {

void transpose_kernel(int dType, void *input, void *output, int nDims, int size,
                      SmallArray strides, SmallArray outputShape);
void transposeSpecial_kernel(int dType, void *input, void *output, int size);
}; // namespace infini

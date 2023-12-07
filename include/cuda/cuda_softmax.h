#pragma once
#include "utils/small_array.h"
namespace infini {
void softmax_kernel(int num_blocks, float *input, float *output, int size,
                    int dimsize, int stride);

} // namespace infini

#pragma once
#include "utils/small_array.h"
namespace infini {
template <typename Tdata>
void softmax_kernel(int num_blocks, Tdata *input, Tdata *output, int size,
                    int dimsize, int stride);
void softmax_stride1_kernel(int num_blocks, float *input, float *output,
                            int size, int dimsize, int stride);
} // namespace infini

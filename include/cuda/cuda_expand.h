#pragma once

#include "operators/unary.h"
#include "utils/small_array.h"
namespace infini {
void expandKernel(int dType, void *input, void *output, int a0, int a1, int a2,
                  int a3, int b0, int b1, int b2, int b3);

void expandRowKernel(int dType, void *input, void *output, int n_rows,
                     int row_len);
}; // namespace infini

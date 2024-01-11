#pragma once

#include "operators/rope.h"
#include "utils/small_array.h"

namespace infini {

void rope_kernel(int dType, int* pos, void *input, void *output, int size, int dim_model, int dim_head, int hidden_stride, int pos_stride);

}; // namespace infini

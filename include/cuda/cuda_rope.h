#pragma once

#include "operators/rope.h"
#include "utils/small_array.h"

namespace infini {

void rope_kernel(int dType, int64_t *pos, void *input, void *output,
                 int dim_model, int dim_head, int batchsize, int pos_stride);

}; // namespace infini

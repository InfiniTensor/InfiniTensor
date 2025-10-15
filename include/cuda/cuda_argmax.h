#pragma once
#include "core/common.h"
#include "core/data_type.h"
#include "utils/small_array.h"
#include <cstdio>
namespace infini {
void argmax_kernel(void *input, int64_t *output, int outer, int inner,
                   int axis_size, int selectLastIndex, int dtype);
} // namespace infini
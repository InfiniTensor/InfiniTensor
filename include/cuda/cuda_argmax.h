#pragma once  
#include "core/common.h"  
#include <cstdio>
#include "utils/small_array.h"
#include "core/data_type.h"
namespace infini {  
    void argmax_kernel(void * input, int64_t *output, const int * inputShape, int ndim,
                      int axis, bool keepdims, bool selectLastIndex, DataType dtype);  
} // namespace infini
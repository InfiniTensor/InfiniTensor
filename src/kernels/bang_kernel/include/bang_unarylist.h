#pragma once
#include "cnnl.h"

namespace infini {

typedef enum {
    Abs = 1,
    Relu = 2,
    Sigmoid = 3,
} UnaryOpType;

void unary_kernel_list(cnnlHandle_t handle, const float *input, float *output,
                       const uint32_t num, const uint32_t op_num,
                       int* list);

}; // namespace infini

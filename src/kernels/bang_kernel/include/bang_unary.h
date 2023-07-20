#pragma once
#include "cnnl.h"

namespace infini {

typedef enum {
  Abs     = 1,
  Relu    = 2,
  Sigmoid = 3,
} UnaryOpType;

void unary_kernel(cnnlHandle_t handle, const float *input,
                  float *output, const uint32_t num, const uint32_t op_num, UnaryOpType list[]);

__mlu_global__ void MLUUnaryKernelUnion1(float *output, float *input,
                                         uint32_t num, uint32_t op_list);

}; // namespace infini

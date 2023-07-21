#pragma once
#include "bang/bang_runtime.h"
#include "bang_unarylist.h"
#include "operators/unary.h"

namespace infini {
  // void unary_kernel(cnnlHandle_t handle,
  //                   const float *input,
  //                   float *output,
  //                   const uint32_t num,
  //                   const uint32_t op_num,
  //                   int* list);

  void bang_unary_kernel(const RuntimeObj* obj, const Operator &_op) {
    auto op = as<UnaryKernelObj>(_op);
    float *const aData = (op->getInputs(0)->getRawDataPtr<float *>());
    float *const cData = (op->getOutput()->getRawDataPtr<float *>());

    auto dim = op->getInputs(0)->getDims();
    auto context = dynamic_cast<const BangRuntimeObj *>(obj);
    auto list = op->getOpList();
    int n = dim[0], c = dim[1], h = dim[2], w = dim[3];
    unary_kernel_list(context->cnnlHandle(), aData, cData, n * c * h * w, list.size(), list.data());

  }
}; // namespace infini

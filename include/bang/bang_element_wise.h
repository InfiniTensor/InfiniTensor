#pragma once
#include "bang/bang_runtime.h"
#include "bang_div.h"
#include "operators/element_wise.h"
namespace infini {

void element_wise_kernel(const RuntimeObj *obj, const Operator &_op) {
    auto op = as<ElementWiseObj>(_op);
    float *const aData = (op->getInputs(0)->getRawDataPtr<float *>());
    float *const bData = (op->getInputs(1)->getRawDataPtr<float *>());
    float *const cData = (op->getOutput()->getRawDataPtr<float *>());

    auto dim = op->getInputs(0)->getDims();
    auto context = dynamic_cast<const BangRuntimeObj *>(obj);
    int n = dim[0], c = dim[1], h = dim[2], w = dim[3];
    if (op->getOpType() == OpType::Div)
        div_kernel(context->cnnlHandle(), aData, bData, cData, n * c * h * w);
    else
        IT_TODO_HALT();
}

}; // namespace infini

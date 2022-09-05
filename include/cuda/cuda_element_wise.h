#pragma once

#include "operators/element_wise.h"

namespace infini {
void div_kernel(float *a, float *b, float *c, int num);
void pow_kernel(float *a, float *b, float *c, int num);

void element_wise_kernel(const Operator &_op) {
    auto op = as<ElementWiseObj>(_op);
    float *const aData = (op->getInputs(0)->getRawDataPtr<float *>());
    float *const bData = (op->getInputs(1)->getRawDataPtr<float *>());
    float *const cData = (op->getOutput()->getRawDataPtr<float *>());

    auto dim = op->getInputs(0)->getDims();
    int n = dim[0], c = dim[1], h = dim[2], w = dim[3];
    if (op->getOpType() == OpType::Div)
        div_kernel(aData, bData, cData, n * c * h * w);
    else if (op->getOpType() == OpType::Pow)
        pow_kernel(aData, bData, cData, n * c * h * w);
    else
        IT_TODO_HALT();
}

}; // namespace infini
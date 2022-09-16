#pragma once

#include "operators/element_wise.h"

namespace infini {
void div_kernel(float *a, float *b, float *c, int a0, int a1, int a2, int a3,
                                              int b0, int b1, int b2, int b3,
                                              int c0, int c1, int c2, int c3);
void pow_kernel(float *a, float *b, float *c, int a0, int a1, int a2, int a3,
                                              int b0, int b1, int b2, int b3,
                                              int c0, int c1, int c2, int c3);

void element_wise_kernel(const Operator &_op) {
    auto op = as<ElementWiseObj>(_op);
    float *const aData = (op->getInputs(0)->getRawDataPtr<float *>());
    float *const bData = (op->getInputs(1)->getRawDataPtr<float *>());
    float *const cData = (op->getOutput()->getRawDataPtr<float *>());

    auto aDim = op->getInputs(0)->getDims();
    auto bDim = op->getInputs(1)->getDims();
    auto cDim = op->getOutput()->getDims();
    if (op->getOpType() == OpType::Div)
        div_kernel(aData, bData, cData, aDim[0], aDim[1], aDim[2], aDim[3],
                                        bDim[0], bDim[1], bDim[2], bDim[3],
                                        cDim[0], cDim[1], cDim[2], cDim[3]);
    else if (op->getOpType() == OpType::Pow)
        pow_kernel(aData, bData, cData, aDim[0], aDim[1], aDim[2], aDim[3],
                                        bDim[0], bDim[1], bDim[2], bDim[3],
                                        cDim[0], cDim[1], cDim[2], cDim[3]);
    else
        IT_TODO_HALT();
}

}; // namespace infini

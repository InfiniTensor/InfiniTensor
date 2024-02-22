#pragma once
#include "bang/bang_runtime.h"
#include "bang_highSoftmax.h"
#include "operators/softmax.h"
namespace infini {

void softmax_kernel(const RuntimeObj *obj, const Operator &_op) {
    auto op = as<SoftmaxObj>(_op);
    void *const mlu_src = (op->getInputs(0)->getRawDataPtr<void *>());
    void *const mlu_destination = (op->getOutput()->getRawDataPtr<void *>());

    auto context = dynamic_cast<const BangRuntimeObj *>(obj);
    auto shape = op->getInputs(0)->getDims();
    int nDim = shape.size();
    int axis = op->getAxis();
    int stride = 1;
    int dimsize = shape[axis];
    int num = 1;
    int othersize = 1;
    int frontsize = 1;

    for (int s = nDim - 1; s >= 0; s--) {
        num *= shape[s];
        if (s > axis) {
            stride *= shape[s];
        }
        if (s < axis) {
            frontsize *= shape[s];
        }
        if (s != axis) {
            othersize *= shape[s];
        }
    }
    if (op->getOpType() == OpType::Softmax)
        softmaxKernel(context->cnnlHandle(), (float *)mlu_destination,
                      (float *)mlu_src, nDim, axis, othersize, frontsize,
                      dimsize, stride);
    else
        IT_TODO_HALT();
}

}; // namespace infini

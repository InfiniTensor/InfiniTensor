#pragma once

#include "operators/unary.h"

namespace infini {
void softmax_kernel(float *input, float *output, int num);
void relu_kernel(float *input, float *output, int num);
void sigmoid_kernel(float *input, float *output, int num);
void tanh_kernel(float *input, float *output, int num);
void abs_kernel(float *input, float *output, int num);

void unary_kernel(const Operator &_op) {
    auto op = as<UnaryObj>(_op);
    float *const inputData = (op->getInputs(0)->getRawDataPtr<float *>());
    float *const outputData = (op->getOutput()->getRawDataPtr<float *>());

    auto dim = op->getInputs(0)->getDims();
    int n = dim[0], c = dim[1], h = dim[2], w = dim[3];
    if (op->getOpType() == OpType::Softmax)
        softmax_kernel(inputData, outputData, n * c * h * w);
    else if (op->getOpType() == OpType::Relu)
        relu_kernel(inputData, outputData, n * c * h * w);
    else if (op->getOpType() == OpType::Sigmoid)
        sigmoid_kernel(inputData, outputData, n * c * h * w);
    else if (op->getOpType() == OpType::Tanh)
        tanh_kernel(inputData, outputData, n * c * h * w);
    else if (op->getOpType() == OpType::Abs)
        abs_kernel(inputData, outputData, n * c * h * w);
    else
        IT_TODO_HALT();
}

}; // namespace infini

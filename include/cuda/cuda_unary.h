#pragma once

#include "operators/unary.h"

namespace infini {
template <typename T> void softmax_kernel(T *input, T *output, size_t num);
template <typename T> void relu_kernel(T *input, T *output, size_t num);
template <typename T> void sigmoid_kernel(T *input, T *output, size_t num);
template <typename T> void tanh_kernel(T *input, T *output, size_t num);
template <typename T> void abs_kernel(T *input, T *output, size_t num);
template <typename T> void sqrt_kernel(T *input, T *output, size_t num);
template <typename T> void neg_kernel(T *input, T *output, size_t num);
template <typename T> void gelu_kernel(T *input, T *output, size_t num);
template <typename T> void erf_kernel(T *input, T *output, size_t num);
template <typename T> void hard_sigmoid_kernel(T *input, T *output, size_t num);
template <typename T> void hard_swish_kernel(T *input, T *output, size_t num);

void unary_kernel(const Operator &_op);

}; // namespace infini

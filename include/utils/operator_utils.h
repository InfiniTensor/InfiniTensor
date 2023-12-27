#pragma once
#ifndef OPERATOR_UTIL_H
#define OPERATOR_UTIL_H

#include "core/operator.h"
#include "core/tensor.h"

namespace infini {

// Launch a broadcast shape based on the shape of input A and B
Shape infer_broadcast(const Shape &A, const Shape &B);
// Launch the real axis based on rank and current axis
int get_real_axis(const int &axis, const int &rank);
// Check if tensor B is unidirectional broadcastable to tensor A
bool is_unidirectional_broadcasting(const Shape &A, const Shape &B);
// Locate the index with size from Shape
Shape locate_index(size_t, Shape const &);
// Delocate the ShapeIndex from Shape with broadcast
size_t delocate_index(Shape const &, vector<int> const &, vector<int> const &);
// Convert KernelAttrs to a string representation
std::string get_kernel_attrs_str(const KernelAttrs &kernelAttrs);
} // namespace infini

#endif

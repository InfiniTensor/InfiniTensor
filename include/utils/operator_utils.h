#pragma once
#ifndef OPERATOR_UTIL_H
#define OPERATOR_UTIL_H

#include "core/operator.h"
#include "core/tensor.h"

#include "utils/small_array.h"
#include <numeric>

namespace infini {

// Launch a broadcast shape based on the shape of input A and B
Shape infer_broadcast(const Shape &A, const Shape &B);
// Launch the real axis based on rank and current axis
int get_real_axis(const int &axis, const int &rank);
// Check if tensor B is unidirectional broadcastable to tensor A
bool is_unidirectional_broadcasting(const Shape &A, const Shape &B);
// Locate the index with size from Shape
Shape locate_index(size_t inputN, const Shape &shape);
// Delocate the ShapeIndex from Shape with broadcast
size_t delocate_index(const Shape &shapeIndex, const Shape &shape,
                      const Shape &stride);
// Convert KernelAttrs to a string representation
std::string get_kernel_attrs_str(const KernelAttrs &kernelAttrs);
// VectorProd
int shapeProd(std::vector<int>::iterator start, std::vector<int>::iterator end);
void broadcastShape(const Shape &originShape, SmallArray &modifyShape,
                    int nDims, int size);
void broadcastShape(const Shape &tempShape, Shape &modifyShape);

} // namespace infini

#endif

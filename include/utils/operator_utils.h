#pragma once
#ifndef OPERATOR_UTIL_H
#define OPERATOR_UTIL_H

#include "core/graph.h"
#include "core/tensor.h"
#include "frontend/graph.h"

namespace infini {

// Launch a broadcast shape based on the shape of input A and B
Shape infer_broadcast(const Shape &A, const Shape &B);
// Launch the real axis based on rank and current axis
int get_real_axis(const int &axis, const int &rank);

// transform RefactorGraph node to InfiniTensorGraph operator
void addOperatorFromGraphTopo(
    GraphObj &g, refactor::frontend::Operator const &nodeInfo,
    refactor::common::slice_t<size_t> input,
    refactor::common::range_t<size_t> output,
    std::unordered_map<size_t, Tensor> &edgeToTensor,
    std::vector<refactor::frontend::Edge> const &edges);
} // namespace infini

#endif

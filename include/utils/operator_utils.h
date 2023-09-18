#pragma once
#ifndef OPERATOR_UTIL_H
#define OPERATOR_UTIL_H

#include "core/tensor.h"
#include "core/graph.h"
#include "computation/graph.h"

namespace infini {

// Launch a broadcast shape based on the shape of input A and B
Shape infer_broadcast(const Shape &A, const Shape &B);
// Launch the real axis based on rank and current axis
int get_real_axis(const int &axis, const int &rank);

// transform RefactorGraph node to InfiniTensorGraph operator
void addOperatorFromGraphTopo(GraphObj &g,
				std::shared_ptr<refactor::computation::Operator> nodeInfo,
				std::vector<size_t> input, std::vector<size_t> output,
				std::unordered_map<size_t, Tensor> &edgeToTensor,
				std::vector<refactor::computation::Edge> edges);

void addEdgeToTensor(GraphObj &g, size_t index,
				std::shared_ptr<refactor::computation::Tensor> tensor,
				std::unordered_map<size_t, Tensor> &edgeToTensor,
				Runtime runtime);
} // namespace infini

#endif

#pragma once
#ifndef OPERATOR_UTIL_H
#define OPERATOR_UTIL_H

#include "core/graph.h"
#include "core/tensor.h"
#include "graph/graph.h"
#include "operators/batch_norm.h"
#include "operators/concat.h"
#include "operators/conv.h"
#include "operators/element_wise.h"
#include "operators/gather.h"
#include "operators/matmul.h"
#include "operators/pad.h"
#include "operators/pooling.h"
#include "operators/reduce_mean.h"
#include "operators/reshape.h"
#include "operators/slice.h"
#include "operators/softmax.h"
#include "operators/split.h"
#include "operators/transpose.h"
#include "operators/unary.h"

namespace infini {

// Launch a broadcast shape based on the shape of input A and B
Shape infer_broadcast(const Shape &A, const Shape &B);
// Launch the real axis based on rank and current axis
int get_real_axis(const int &axis, const int &rank);

using RefactorNodeInfoCell = refactor::graph::Cell<refactor::graph::NodeInfo>;
using RefactorEdgeInfoCell = refactor::graph::Cell<refactor::graph::EdgeInfo>;
using EdgeRef = GraphTopo<RefactorNodeInfoCell, RefactorEdgeInfoCell>::EdgeRef;
void addOperatorFromGraphTopo(
    GraphObj &g,
    const GraphTopoSearcher<refactor::graph::NodeInfo,
                            refactor::graph::EdgeInfo>::Node node,
    const std::unordered_map<int32_t, Tensor> edgeIdxToTensor,
    const std::unordered_map<int32_t, infini::Shape> edgeIdxToShape);
RefactorNodeInfoCell getNodeInfo(const Operator &obj);
void processShapeVariable(
    const Operator &obj,
    GraphTopo<RefactorNodeInfoCell, RefactorEdgeInfoCell> &graphTopo,
    std::vector<EdgeRef> &nodeInputs);
} // namespace infini

#endif

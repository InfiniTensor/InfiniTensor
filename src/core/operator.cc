#include "core/operator.h"
#include "core/graph.h"
#include "core/hash.h"

namespace infini {

bool OperatorNode::isLinearOp() const {
    return enum_to_underlying(type) >= 100 && enum_to_underlying(type) < 200;
}

bool OperatorNode::isElementWiseOp() const {
    return enum_to_underlying(type) >= 200 && enum_to_underlying(type) < 300;
}

bool OperatorNode::isSplitOp() const { return type == OpType::Split; }

bool OperatorNode::isConcatOp() const { return type == OpType::Concat; }

bool OperatorNode::isComputeOp() const {
    return type == OpType::Conv || type == OpType::Matmul ||
           type == OpType::ConvTrans || type == OpType::G2BMM ||
           type == OpType::GBMML;
}

bool OperatorNode::isTransposeOp() const { return type == OpType::Transpose; }

bool OperatorNode::isReshapeOp() const { return type == OpType::Reshape; }

bool OperatorNode::isMemBoundOp() const {
    return type == OpType::MemBound || type == OpType::Activation ||
           type == OpType::Transpose;
}

OpPerfKey OperatorNode::getOpPerfKey() const {
    auto workloadVector = getWorkloadVector();
    // Calculate hash of workload, i.e. hash with shape. This is different from
    // Operator::hash, which hashes operator attributes and ignores tensor
    // shapes.
    HashType hash = 0;
    hash = hashAppend(hash, enum_to_underlying(type));
    hash = hashAppend(hash, hashVector(workloadVector));
    return OpPerfKey(hash, type, workloadVector);
}

HashType OperatorNode::hash() const {
    HashType hash = 0;
    hash = hashAppend(hash, enum_to_underlying(type));
    hash = hashAppend(hash, hashVector(getOpAttrVector()));
    return hash;
}

bool OperatorNode::checkValid(GraphNode *graph) {
    auto optShapes = inferShape();
    if (!optShapes) // shape inference failed
        return false;
    const vector<Shape> &shapes = *optShapes;
    if (shapes.size() != outputs.size())
        return false;
    if (graph) { // if graph != nullptr, outputs should be created
        for (size_t i = 0; i < outputs.size(); i++) {
            IT_ASSERT(!outputs[i]);
            outputs[i] = graph->addTensor(shapes[i]);
        }
    } else { // if graph is not empty, check outputs match inferred shapes
        for (size_t i = 0; i < shapes.size(); ++i) {
            if (shapes[i] != outputs[i]->getDims())
                return false;
        }
    }
    return true;
}

optional<vector<Shape>> OperatorNode::inferShape() const {
    return inferShape(inputs);
}

} // namespace infini
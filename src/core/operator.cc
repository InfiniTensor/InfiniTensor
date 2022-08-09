#include "core/operator.h"

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

} // namespace infini
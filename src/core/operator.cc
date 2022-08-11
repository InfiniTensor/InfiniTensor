#include "core/operator.h"
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

bool OperatorNode::checkValid() const {
    if (auto optVecShape = inferShape()) {
        if (!optVecShape)
            return false;
        if (optVecShape->size() != outputs.size())
            return false;
        for (size_t i = 0; i < optVecShape->size(); ++i) {
            if ((*optVecShape)[i] != outputs.at(i)->getDims())
                return false;
        }
    }
    return true;
}

} // namespace infini
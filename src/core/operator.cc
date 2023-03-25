#include "core/operator.h"
#include "core/graph.h"
#include "core/hash.h"

namespace infini {

OperatorObj::OperatorObj(OpType opType, TensorVec inputs, TensorVec outputs)
    : type(opType), inputs(inputs), outputs(outputs) {
    for (const auto &t : inputs)
        IT_ASSERT(t);
}

OperatorObj::OperatorObj(OpType opType) : type(opType) {}

bool OperatorObj::isLinearOp() const {
    return enum_to_underlying(type) >= 100 && enum_to_underlying(type) < 200;
}

bool OperatorObj::isElementWiseOp() const {
    return enum_to_underlying(type) >= 200 && enum_to_underlying(type) < 300;
}

bool OperatorObj::isSplitOp() const { return type == OpType::Split; }

bool OperatorObj::isConcatOp() const { return type == OpType::Concat; }

bool OperatorObj::isComputeOp() const {
    return type == OpType::Conv || type == OpType::Matmul ||
           type == OpType::ConvTrans || type == OpType::ConvTransNHWC ||
           type == OpType::G2BMM || type == OpType::GBMM;
}

bool OperatorObj::isTransposeOp() const { return type == OpType::Transpose; }

bool OperatorObj::isReshapeOp() const { return type == OpType::Reshape; }

bool OperatorObj::isMemBoundOp() const {
    return type == OpType::MemBound || type == OpType::Activation ||
           type == OpType::Transpose;
}

void OperatorObj::removePredecessors(const Operator &op) {
    for (auto it = predecessors.begin(); it != predecessors.end();) {
        if (it->lock() == op)
            it = predecessors.erase(it);
        else
            ++it;
    }
}

void OperatorObj::removeSuccessors(const Operator &op) {
    for (auto it = successors.begin(); it != successors.end();) {
        if (it->lock() == op)
            it = successors.erase(it);
        else
            ++it;
    }
}

void OperatorObj::replaceInput(Tensor t1, Tensor t2) {
    for (auto itr = inputs.begin(); itr != inputs.end(); ++itr) {
        if (*itr == t1) {
            *itr = t2;
        }
    }
}

OpPerfKey OperatorObj::getOpPerfKey() const {
    auto workloadVector = getWorkloadVector();
    // Calculate hash of workload, i.e. hash with shape. This is different from
    // Operator::hash, which hashes operator attributes and ignores tensor
    // shapes.
    HashType hash = 0;
    hash = hashAppend(hash, enum_to_underlying(type));
    hash = hashAppend(hash, hashVector(workloadVector));
    return OpPerfKey(hash, type, workloadVector);
}

HashType OperatorObj::hash() const {
    HashType hash = 0;
    hash = hashAppend(hash, enum_to_underlying(type));
    hash = hashAppend(hash, hashVector(getOpAttrVector()));
    return hash;
}

bool OperatorObj::checkValid(GraphObj *graph) {
    auto optShapes = inferShape();
    if (!optShapes) // shape inference failed
        return false;

    const vector<Shape> &shapes = *optShapes;
    if (shapes.size() != outputs.size())
        return false;
    if (graph) { // if graph != nullptr, outputs should be created
        auto dataTypes = inferDataType();
        for (size_t i = 0; i < outputs.size(); i++) {
            IT_ASSERT(!outputs[i], "Find empty output while operator creation");
            outputs[i] = graph->addTensor(shapes[i], dataTypes[i]);
        }
    } else { // if outputs have been created, check their shapes
        for (size_t i = 0; i < shapes.size(); ++i) {
            if (shapes[i] != outputs[i]->getDims())
                return false;
        }
    }
    return true;
}

bool OperatorObj::checkValid(GraphObj *graph, DataType type) {
    auto optShapes = inferShape();
    if (!optShapes) // shape inference failed
        return false;

    const vector<Shape> &shapes = *optShapes;
    if (shapes.size() != outputs.size())
        return false;
    if (graph) { // if graph != nullptr, outputs should be created
        auto dataTypes = vector(numOutputs(), type);
        for (size_t i = 0; i < outputs.size(); i++) {
            IT_ASSERT(!outputs[i]);
            outputs[i] = graph->addTensor(shapes[i], dataTypes[i]);
        }
    } else { // if outputs have been created, check their shapes
        for (size_t i = 0; i < shapes.size(); ++i) {
            if (shapes[i] != outputs[i]->getDims())
                return false;
        }
    }
    return true;
}

optional<vector<Shape>> OperatorObj::inferShape() const {
    return inferShape(inputs);
}

vector<DataType> OperatorObj::inferDataType(const TensorVec &inputs) const {
    auto dataType = inputs[0]->getDType();
    for (const auto &tensor : inputs)
        IT_ASSERT(dataType == tensor->getDType());
    return vector(numOutputs(), dataType);
}

vector<DataType> OperatorObj::inferDataType() const {
    return inferDataType(inputs);
}

} // namespace infini

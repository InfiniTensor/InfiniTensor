#include "core/operator.h"
#include "core/graph.h"
#include "core/hash.h"
#include <algorithm>

namespace infini {

OperatorObj::OperatorObj(OpType opType, TensorVec inputs, TensorVec outputs)
    : type(opType), inputs(inputs), outputs(outputs) {
    if (opType != OpType::Recv) {
        for (const auto &t : inputs)
            IT_ASSERT(t);
    }
}

vector<DimSource> OperatorObj::dimSources(size_t output, size_t dim) const {
    IT_ASSERT(output < outputs.size());
    IT_ASSERT(dim < outputs[output]->getRank());
    // Knowing nothing about how this operator works out its dimensions, every
    // dimension of every input has to count as one the output follows.
    vector<DimSource> sources;
    for (size_t i = 0; i < inputs.size(); ++i) {
        for (size_t d = 0; d < inputs[i]->getRank(); ++d) {
            sources.push_back(DimSource{i, d});
        }
    }
    return sources;
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
    hash = hashAppend(hash, type.underlying());
    hash = hashAppend(hash, hashVector(workloadVector));
    return OpPerfKey(hash, type, workloadVector);
}

HashType OperatorObj::hash() const {
    HashType hash = 0;
    hash = hashAppend(hash, type.underlying());
    hash = hashAppend(hash, hashVector(getOpAttrVector()));
    return hash;
}

bool OperatorObj::checkValid(GraphObj *graph) {
    auto optShapes = inferShape();
    if (!optShapes) // shape inference failed
        return false;

    auto shapes = *optShapes;
    if (shapes.size() != outputs.size())
        return false;
    if (graph) { // if graph != nullptr, outputs should be created
        // ONNX import realizes every symbolic dimension as one while the graph
        // is built. Geometry operators can therefore calculate a non-positive
        // placeholder-only output before set_input supplies a real shape.
        // Preserve construction for a declared dynamic input, but do not hide
        // invalid shapes in ordinary static graphs.
        const bool hasDynamicInput =
            std::any_of(inputs.begin(), inputs.end(), [](const Tensor &input) {
                const auto &descs = input->getDimDescs();
                return !descs.empty() && std::any_of(descs.begin(), descs.end(),
                                                     [](const DimDesc &desc) {
                                                         return desc.dynamic;
                                                     });
            });
        // Slice resolves even placeholder inputs to valid nonnegative ranges;
        // an empty slice must not acquire an element during construction.
        if (hasDynamicInput && type != OpType::Slice) {
            for (auto &shape : shapes) {
                for (auto &dim : shape)
                    dim = std::max(dim, 1);
            }
        }

        auto dataTypes = inferDataType();
        for (size_t i = 0; i < outputs.size(); i++) {
            IT_ASSERT(!outputs[i], "Find empty output while operator creation");
            outputs[i] = graph->addTensor(shapes[i], dataTypes[i]);
            // Outputs are created before GraphObj::shape_infer can derive their
            // precise sources. Keep the dynamic marker moving through eager
            // construction so a later geometry operator gets the same guard.
            if (hasDynamicInput) {
                outputs[i]->setDimDescs(
                    DimDescs(shapes[i].size(), DimDesc{true, ""}));
            }
        }
    } else { // if outputs have been created, check their shapes
        for (size_t i = 0; i < shapes.size(); ++i) {
            if (shapes[i] != outputs[i]->getDims())
                return false;
        }
    }
    // The outputs exist by now, so their contents can be worked out.
    inferShapeValue();
    return true;
}

optional<vector<Shape>> OperatorObj::inferShape() { return inferShape(inputs); }

vector<DataType> OperatorObj::inferDataType(const TensorVec &inputs) const {
    auto dataType = inputs[0]->getDType();
    return vector(numOutputs(), dataType);
}

vector<DataType> OperatorObj::inferDataType() const {
    return inferDataType(inputs);
}

bool OperatorObj::beginShapeValueUpdate() {
    for (const auto &output : outputs) {
        output->clearShapeValue();
    }
    for (const auto &output : outputs) {
        if (!output->canHoldShapeValue()) {
            return false;
        }
    }
    for (const auto &input : inputs) {
        if (!input->getShapeValue().has_value()) {
            return false;
        }
    }
    return true;
}

} // namespace infini

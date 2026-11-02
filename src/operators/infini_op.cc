#include "operators/infini_op.h"
#include "core/graph.h"
namespace infini {
TensorVec
InfiniOpObj::flatten(const std::map<std::string, TensorVec> &bindings) {
    TensorVec result;
    for (const auto &[name, tensors] : bindings) {
        IT_ASSERT(!name.empty() && !tensors.empty(),
                  "Empty InfiniOp tensor binding");
        result.insert(result.end(), tensors.begin(), tensors.end());
    }
    return result;
}
InfiniOpObj::InfiniOpObj(GraphObj *, std::string operation,
                         const std::map<std::string, TensorVec> &inputMap,
                         const std::map<std::string, TensorVec> &outputMap,
                         nlohmann::json attributes)
    : OperatorObj(OpType::InfiniOp, flatten(inputMap), flatten(outputMap)),
      operation(std::move(operation)), attributes(std::move(attributes)) {
    IT_ASSERT(!this->operation.empty() && this->attributes.is_object(),
              "Invalid InfiniOp definition");
    IT_ASSERT(!outputs.empty(), "InfiniOp requires explicit output tensors");
    size_t index = 0;
    for (const auto &[name, tensors] : inputMap) {
        IT_ASSERT(!outputMap.count(name) && !this->attributes.contains(name),
                  "Duplicate InfiniOp argument");
        inputBindings[name] = {index, tensors.size()};
        index += tensors.size();
    }
    index = 0;
    for (const auto &[name, tensors] : outputMap) {
        IT_ASSERT(!this->attributes.contains(name),
                  "Duplicate InfiniOp argument");
        outputBindings[name] = {index, tensors.size()};
        index += tensors.size();
        for (const auto &tensor : tensors) {
            IT_ASSERT(tensor != nullptr);
            outputShapes.push_back(tensor->getDims());
        }
    }
    for (const auto &tensor : inputs)
        inputShapes.push_back(tensor->getDims());
    IT_ASSERT(checkValid(nullptr));
}
optional<vector<Shape>> InfiniOpObj::inferShape(const TensorVec &values) {
    IT_ASSERT(values.size() == inputShapes.size());
    if (acceptNextInputShapes) {
        for (size_t i = 0; i < values.size(); ++i)
            inputShapes[i] = values[i]->getDims();
        acceptNextInputShapes = false;
    }
    for (size_t i = 0; i < values.size(); ++i)
        IT_ASSERT(values[i]->getDims() == inputShapes[i],
                  "InfiniOp input shape changed; supply explicit output shapes "
                  "before shape inference");
    return outputShapes;
}
void InfiniOpObj::setOutputShapes(vector<Shape> shapes) {
    IT_ASSERT(shapes.size() == outputs.size(),
              "InfiniOp output shape count mismatch");
    for (const auto &shape : shapes)
        for (auto dim : shape)
            IT_ASSERT(dim >= 0, "InfiniOp output shape must be concrete");
    outputShapes = std::move(shapes);
    // Upstream intermediate shapes will be updated by the next topology walk.
    acceptNextInputShapes = true;
}
std::string InfiniOpObj::toString() const {
    return "InfiniOp[" + operation + "]" + attributes.dump();
}
vector<int> InfiniOpObj::getOpAttrVector() const {
    const auto text = toString();
    return vector<int>(text.begin(), text.end());
}
vector<int> InfiniOpObj::getWorkloadVector() const {
    auto result = getOpAttrVector();
    for (const auto &tensor : inputs) {
        const auto shape = tensor->getDims();
        result.push_back(shape.size());
        result.insert(result.end(), shape.begin(), shape.end());
    }
    return result;
}
} // namespace infini

#pragma once
#include "core/operator.h"
#include <map>
#include <nlohmann/json.hpp>

namespace infini {
// Shapes and dtypes are supplied by the frontend. No shape semantics are
// guessed from an operator's C++ parameter types or from its name.
class InfiniOpObj final : public OperatorObj {
    std::string operation;
    std::map<std::string, std::pair<size_t, size_t>> inputBindings,
        outputBindings;
    nlohmann::json attributes;
    vector<Shape> inputShapes, outputShapes;
    bool acceptNextInputShapes = false;
    static TensorVec flatten(const std::map<std::string, TensorVec> &bindings);

  public:
    InfiniOpObj(GraphObj *graph, std::string operation,
                const std::map<std::string, TensorVec> &inputs,
                const std::map<std::string, TensorVec> &outputs,
                nlohmann::json attributes = nlohmann::json::object());
    OP_CLONE(InfiniOpObj);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;
    std::string toString() const override;
    int numInputs() const override { return inputs.size(); }
    int numOutputs() const override { return outputs.size(); }
    const std::string &getOperation() const { return operation; }
    const auto &getInputBindings() const { return inputBindings; }
    const auto &getOutputBindings() const { return outputBindings; }
    const auto &getAttributes() const { return attributes; }
    void setOutputShapes(vector<Shape> shapes);

  private:
    vector<int> getOpAttrVector() const override;
    vector<int> getWorkloadVector() const override;
};
} // namespace infini

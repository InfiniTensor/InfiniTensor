#pragma once
#include "core/operator.h"

namespace infini {
class RoPEObj : public OperatorObj {
  public:
    /**
     * @brief Construct a new RotaryEmbedding object.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param pos The positon id of the query.
     * @param input The input tensor.
     * @param output The output tensor.
     */
    RoPEObj(GraphObj *graph, Tensor pos, Tensor input, Tensor output);
    OP_CLONE(RoPEObj);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;

    std::string toString() const override;
    int numInputs() const override { return 2; }
    int numOutputs() const override { return 1; }
    DataType getDType() const { return getInputs(1)->getDType(); }
    
    vector<DataType> inferDataType(const TensorVec &inputs) const override {
        return {inputs[1]->getDType()};
    };

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

} // namespace infini

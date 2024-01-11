#pragma once
#include "core/operator.h"

namespace infini {
class RoPEObj : public OperatorObj {
  public:
    RoPEObj(GraphObj *graph, Tensor pos, Tensor input, Tensor output);
    OP_CLONE(RoPEObj);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;

    std::string toString() const override;
    int numInputs() const override { return 2; }
    int numOutputs() const override { return 1; }
    DataType getDType() const { return getInputs(1)->getDType(); }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

} // namespace infini

#pragma once
#include "core/operator.h"

namespace infini {
class SwiGLUObj : public OperatorObj {
  public:
    SwiGLUObj(GraphObj *graph, Tensor input, Tensor gate, Tensor output);
    OP_CLONE(SwiGLUObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;

    std::string toString() const override;
    int numInputs() const override { return 2; }
    int numOutputs() const override { return 1; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};
} // namespace infini

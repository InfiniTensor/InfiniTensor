#pragma once
#include "core/operator.h"

namespace infini {
class DetObj : public OperatorObj {
  public:
    DetObj(GraphObj *graph, Tensor input, Tensor output);
    OP_CLONE(DetObj);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;

    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};
}; // namespace infini

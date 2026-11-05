#pragma once

#include "core/operator.h"

namespace infini {

class MaskedScatterObj : public OperatorObj {
  public:
    MaskedScatterObj(GraphObj *graph, Tensor input, Tensor mask, Tensor source, Tensor output);
    OP_CLONE(MaskedScatterObj);
    int numInputs() const override { return 3; }
    int numOutputs() const override { return 1; }
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;
    std::string toString() const override;

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

} // namespace infini

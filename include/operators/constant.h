#pragma once
#include "core/operator.h"

namespace infini {

class ConstantObj : public OperatorObj {
  Tensor value;
  public:
    
    ConstantObj(GraphObj *graph,
             Tensor output, Tensor value);
    OP_CLONE(ConstantObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    std::string toString() const override;
    int numInputs() const override { return 0; }
    int numOutputs() const override { return 1; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

} // namespace infini
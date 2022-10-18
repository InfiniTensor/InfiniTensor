#pragma once
#include "core/operator.h"

namespace infini {

class BiasPReLU : public OperatorObj {
  protected:
    bool PReLU;
    float paramPReLU;

    int n, h, w, c;

  public:
    BiasPReLU(GraphObj *graph, Tensor input, Tensor bias, Tensor output,
              bool PReLU_, float paramPReLU_);

    std::string toString() const override { return "Bias2PReLU"; }
    int numInputs() const override { return 2; }
    int numOutputs() const override { return 1; }

    bool getPReLU() const { return PReLU; }
    float getParamReLU() const { return paramPReLU; }
    Tensor getBias() const { return inputs[1]; }

    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

} // namespace infini
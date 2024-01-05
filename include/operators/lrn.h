#pragma once
#include "core/operator.h"

namespace infini {
class LRNObj : public OperatorObj {

  public:
    LRNObj(GraphObj *graph, Tensor inputX, Tensor inputY, float alpha,
           float beta, float bias, int size);
    OP_CLONE(LRNObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;

    std::string toString() const override;
    int numInputs() const override { return inputs.size(); }
    int numOutputs() const override { return 1; }
    auto getAlphaBetaBias() const {
        return tuple(alpha_value, beta_value, bias_value);
    }
    auto getSize() const { return size_value; }

  private:
    float alpha_value, beta_value, bias_value;
    int size_value;
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

} // namespace infini

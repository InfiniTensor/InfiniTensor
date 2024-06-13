#pragma once
#include "core/operator.h"

namespace infini {
class EluObj : public OperatorObj {

  public:
    EluObj(GraphObj *graph, Tensor input, Tensor output, float alpha);
    OP_CLONE(EluObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;
    std::string toString() const override;
    int numInputs() const override { return inputs.size(); }
    int numOutputs() const override { return 1; }
    float getAlpha() const { return alpha; }
    float alpha;

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};
} // namespace infini

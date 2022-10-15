#pragma once
#include "core/operator.h"

namespace infini {
class ConcatObj : public OperatorObj {
    int dim;

  public:
    ConcatObj(GraphObj *graph, TensorVec inputs, Tensor output, int dim);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    std::string toString() const override;
    int numInputs() const override { return inputs.size(); }
    int numOutputs() const override { return 1; }
    int getDim() const { return dim; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};
} // namespace infini

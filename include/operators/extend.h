#pragma once
#include "core/operator.h"

namespace infini {
class ExtendObj : public OperatorObj {
    int dim, num; // copy num times at the dim.

  public:
    ExtendObj(GraphObj *graph, Tensor input, Tensor output, int dim,
              int num = 1);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }
    int getDim() const { return dim; }
    int getNum() const { return num; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};
} // namespace infini

#pragma once
#include "core/operator.h"

namespace infini {
class SplitObj : public OperatorObj {
    int dim, num;      // split dim;Average split num or outputs size
    vector<int> ratio; // output dim ratio
  public:
    SplitObj(GraphObj *graph, Tensor input, std::optional<TensorVec> outputs,
             int dim, int num);
    SplitObj(GraphObj *graph, Tensor input, std::optional<TensorVec> outputs,
             int dim, const vector<int> &ratio);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return num; }
    int getDim() const { return dim; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};
} // namespace infini
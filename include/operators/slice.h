#pragma once
#include "core/operator.h"

namespace infini {
class SliceObj : public OperatorObj {
    vector<int> starts, ends; // the start no. and end no. for all dims.

  public:
    SliceObj(GraphObj *graph, Tensor input, Tensor output,
             const vector<int> &starts, const vector<int> &ends,
             const optional<vector<int>> &axis,
             const optional<vector<int>> &steps);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;
    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }
    Shape getStart() const { return starts; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};
} // namespace infini
#pragma once

#include "core/operator.h"
#include <tuple>

namespace infini {

class GridSampleObj : public OperatorObj {
  public:
    GridSampleObj(GraphObj *graph, Tensor input, Tensor grid, Tensor output,
                  bool nearest, bool reflection, bool alignCorners);
    OP_CLONE(GridSampleObj);
    int numInputs() const override { return 2; }
    int numOutputs() const override { return 1; }
    auto getAttributes() const { return std::tuple(nearest, reflection, alignCorners); }
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;
    std::string toString() const override;

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
    bool nearest, reflection, alignCorners;
};

} // namespace infini

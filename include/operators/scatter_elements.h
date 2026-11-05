#pragma once

#include "core/operator.h"

namespace infini {

class ScatterElementsObj : public OperatorObj {
  public:
    ScatterElementsObj(GraphObj *graph, Tensor input, Tensor indices, Tensor updates,
                        Tensor output, int axis, std::string reduction);
    OP_CLONE(ScatterElementsObj);
    int numInputs() const override { return 3; }
    int numOutputs() const override { return 1; }
    int getAxis() const { return axis; }
    const std::string &getReduction() const { return reduction; }
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;
    std::string toString() const override;

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
    int axis;
    std::string reduction;
};

} // namespace infini

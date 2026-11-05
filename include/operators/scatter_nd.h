#pragma once

#include "core/operator.h"

namespace infini {

class ScatterNdObj : public OperatorObj {
  public:
    ScatterNdObj(GraphObj *graph, Tensor input, Tensor indices, Tensor updates,
                  Tensor output, bool add);
    OP_CLONE(ScatterNdObj);
    int numInputs() const override { return 3; }
    int numOutputs() const override { return 1; }
    bool isAdd() const { return add; }
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;
    std::string toString() const override;

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
    bool add;
};

} // namespace infini

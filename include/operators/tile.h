#pragma once

#include "core/operator.h"

namespace infini {

class TileObj : public OperatorObj {
  public:
    TileObj(GraphObj *graph, Tensor input, Tensor output, Shape repeats);
    OP_CLONE(TileObj);
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;
    std::string toString() const override;

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
    Shape repeats;
};

} // namespace infini

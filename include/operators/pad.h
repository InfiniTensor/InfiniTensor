#pragma once
#include "core/operator.h"

namespace infini {
class PadObj : public OperatorObj {
    // the number of start and end pad values for all dims.
    vector<int> pads;

  public:
    // pad for appointed axises,if axis is empty,then pad for all axises.
    PadObj(GraphObj *graph, Tensor input, Tensor output,
           const vector<int> &pads, const optional<const vector<int>> &axis);
    OP_CLONE(PadObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;
    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }
    Shape getPads() const { return pads; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};
} // namespace infini

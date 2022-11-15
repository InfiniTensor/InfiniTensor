#pragma once

#include "core/operator.h"

namespace infini {
class GatherObj : public OperatorObj {
    int axis;

  public:
    GatherObj(GraphObj *graph, Tensor input, Tensor index, Tensor output,
              int axis);
    OP_CLONE(GatherObj);
    std::string toString() const override;
    int numInputs() const override { return 2; }
    int numOutputs() const override { return 1; }
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;
    int getAxis() const { return axis; }
    vector<DataType> inferDataType(const TensorVec &inputs) const override;

  private:
    bool CheckIndexValid() const;
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};
} // namespace infini

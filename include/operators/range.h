#pragma once
#include "core/operator.h"

namespace infini {
class RangeObj : public OperatorObj {
  
    float start, limit,delta;

  public:
    RangeObj(GraphObj *graph, float start, float limit, float delta,
                Tensor output);

    OP_CLONE(RangeObj);
    std::string toString() const override;

    int numInputs() const override { return inputs.size(); }
    int numOutputs() const override { return outputs.size(); }
    float getStart() const { return start; };
    float getLimit() const { return limit; };
    float getDelta() const { return delta; };
    vector<DataType> inferDataType(const TensorVec &inputs) const override;
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;

};
} // namespace infini

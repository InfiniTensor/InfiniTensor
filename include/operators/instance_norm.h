#pragma once
#include "core/operator.h"

namespace infini {
class InstanceNormObj : public OperatorObj {
    float eps;

  public:
    InstanceNormObj(GraphObj *graph, Tensor input, Tensor output, Tensor scale,
                    Tensor bias, float eps = 1e-5);
    OP_CLONE(InstanceNormObj);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;
    std::string toString() const override;

    int numInputs() const override { return inputs.size(); }
    int numOutputs() const override { return outputs.size(); }
    float getEps() const { return eps; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;

    vector<DataType> inferDataType(const TensorVec &inputs) const override;
};
} // namespace infini

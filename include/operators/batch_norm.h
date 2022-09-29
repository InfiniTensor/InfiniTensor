#pragma once
#include "core/operator.h"

namespace infini {
class BatchNormObj : public OperatorObj {
    float momentum, eps;
    bool training;

  public:
    BatchNormObj(GraphObj *graph, Tensor input, Tensor output, Tensor mean,
                 Tensor var, Tensor scale, Tensor bias, float momentum = 0.9,
                 float eps = 1e-5, bool training = false);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    std::string toString() const override;

    // output size will be 3 when training
    int numInputs() const override { return 5; }
    int numOutputs() const override { return outputs.size(); }
    float getEps() const { return eps; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;

    vector<DataType> inferDataType(const TensorVec &inputs) const override;
};
} // namespace infini

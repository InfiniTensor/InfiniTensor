#pragma once
#include "core/operator.h"

namespace infini {
class LayerNormObj : public OperatorObj {
    float eps;
    int axis, stash_type;

  public:
    LayerNormObj(GraphObj *graph, Tensor input, Tensor scale, Tensor bias,
                 Tensor output, float eps = 1e-5, int axis = -1,
                 int stash_type = 1);
    OP_CLONE(LayerNormObj);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;
    std::string toString() const override;

    int numInputs() const override { return 5; }
    int numOutputs() const override { return outputs.size(); }
    float getEps() const { return eps; }
    int getAxis() const { return axis; }
    int getStashType() const { return stash_type; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;

    vector<DataType> inferDataType(const TensorVec &inputs) const override;
};
} // namespace infini

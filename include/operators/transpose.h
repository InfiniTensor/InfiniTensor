#pragma once
#include "core/operator.h"

namespace infini {
class TransposeObj : public OperatorObj {
    vector<int> transposePermute;

  public:
    TransposeObj(GraphObj *graph, Tensor input, Tensor output,
                 vector<int> permute);
    OP_CLONE(TransposeObj);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }
    std::vector<int> getPermute() const { return transposePermute; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};
}; // namespace infini

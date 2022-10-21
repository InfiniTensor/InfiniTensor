#pragma once

#include "core/operator.h"

namespace infini {
class TransposeObj : public OperatorObj {
    Shape perm;

  public:
    TransposeObj(GraphObj *graph, Tensor input, Tensor output,
                 const Shape &dims);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }
    Shape getPerm() const { return perm; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

} // namespace infini

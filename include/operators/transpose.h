#pragma once
#include "core/operator.h"

namespace infini {
class TransposeObj : public OperatorObj {
  public:
    TransposeObj(GraphObj *graph, Tensor input, Tensor output, int permute[4]);
    OP_CLONE(TransposeObj);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }
    std::vector<int> getPermute() const {
        return std::vector<int>{
            transposePermute[0],
            transposePermute[1],
            transposePermute[2],
            transposePermute[3],
        };
    }

  private:
    int transposePermute[4];
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};
}; // namespace infini

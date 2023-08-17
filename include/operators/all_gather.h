#pragma once
#include "core/operator.h"

namespace infini {
class AllGatherObj : public OperatorObj {

  public:
    AllGatherObj(GraphObj *graph, Tensor input, Tensor output);
    OP_CLONE(AllGatherObj);
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override {
        return {{inputs[0]->getDims()}};
    };
    std::string toString() const override;

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};
} // namespace infini
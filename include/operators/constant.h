#pragma once

#include "core/operator.h"

namespace infini {
class ConstantObj : public OperatorObj {

  public:
    ConstantObj(GraphObj *graph, Tensor output)
        : OperatorObj(OpType::Constant, {}, {output}) {
        IT_ASSERT(output);
        IT_ASSERT(checkValid(graph));
    }
    OP_CLONE(ConstantObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) const {
        return {{outputs[0]->getDims()}};
    };

    std::string toString() const override;
    int numInputs() const override { return 0; }
    int numOutputs() const override { return 1; }
    void makeConstant() {}

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};
} // namespace infini

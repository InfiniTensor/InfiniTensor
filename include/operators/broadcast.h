#pragma once
#include "core/operator.h"

namespace infini {
class BroadcastObj : public OperatorObj {
  public:
    BroadcastObj(GraphObj *graph, Tensor input, Tensor output, int root);
    OP_CLONE(BroadcastObj);
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override {
        return {{inputs[0]->getDims()}};
    };
    std::string toString() const override;
    int getRoot() const { return root; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
    vector<DataType> inferDataType(const TensorVec &inputs) const override {
        return {inputs[0]->getDType()};
    };

  protected:
    // The rank who broadcasts data among this communication group
    int root;
};

} // namespace infini
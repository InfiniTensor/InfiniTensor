#pragma once
#include "core/operator.h"

namespace infini {
class AllGatherObj : public OperatorObj {

  public:
    AllGatherObj(GraphObj *graph, Tensor input, std::optional<TensorVec>,
                 int world_size);
    OP_CLONE(AllGatherObj);
    int numInputs() const override { return 1; }
    int numOutputs() const override { return world_size; }
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;
    std::string toString() const override;
    int getWorldSize() const { return world_size; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
    vector<DataType> inferDataType(const TensorVec &inputs) const override;

  protected:
    int world_size;
};
} // namespace infini
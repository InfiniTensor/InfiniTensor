#pragma once

#include "core/operator.h"

namespace infini {
/**
 * @brief Gather and concatenate given positions on a certain dimension of the
 * input tensor using an index tensor.
 *
 */
class GatherObj : public OperatorObj {
    int axis;

  public:
    /**
     * @brief Construct a new Gather object.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param input The input tensor.
     * @param indices The index tensor.
     * @param output The output tensor.
     * @param axis The axis to gather on.
     */
    GatherObj(GraphObj *graph, Tensor input, Tensor indices, Tensor output,
              int axis);
    OP_CLONE(GatherObj);
    std::string toString() const override;
    int numInputs() const override { return 2; }
    int numOutputs() const override { return 1; }
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;
    int getAxis() const { return axis; }
    vector<DataType> inferDataType(const TensorVec &inputs) const override;

  private:
    bool CheckIndexValid() const;
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};
} // namespace infini

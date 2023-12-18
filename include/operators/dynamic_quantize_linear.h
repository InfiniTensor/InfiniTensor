#pragma once
#include "core/operator.h"

namespace infini {
/**
 * @brief A Function to fuse calculation for Scale, Zero Point and FP32->8Bit
 * conversion of FP32 Input data.
 *
 */
class DynamicQuantizeLinearObj : public OperatorObj {

  public:
    /**
     * @brief Construct a new DynamicQuantizeLinear object.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param input The input tensor.
     * @param outputs The output tensors.
     */
    DynamicQuantizeLinearObj(GraphObj *graph, Tensor input,
                             std::optional<TensorVec> outputs);
    OP_CLONE(DynamicQuantizeLinearObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;

    std::string toString() const override;
    int numInputs() const override { return inputs.size(); }
    int numOutputs() const override { return 3; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;

    vector<DataType> inferDataType(const TensorVec &inputs) const override;
};

} // namespace infini

#pragma once
#include "core/operator.h"

namespace infini {
/**
 * @brief y = (x - x_zero_point) *x_scale
 *
 */
class DequantizeLinearObj : public OperatorObj {
    int axis;

  public:
    /**
     * @brief Construct a new DequantizeLinear object.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param inputX The input tensor X.
     * @param inputScale The input tensor x_scale.
     * @param output The output tensor.
     * @param inputZeroPoint The z_zero_point.
     */
    DequantizeLinearObj(GraphObj *graph, Tensor inputX, Tensor inputScale,
                        Tensor output, Tensor inputZeroPoint = nullptr,
                        int axis = 1);
    OP_CLONE(DequantizeLinearObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;

    std::string toString() const override;

    Tensor getZeroPoint() const {
        return inputs.size() > 2 ? inputs[2] : nullptr;
    }
    int numInputs() const override { return inputs.size(); }
    int numOutputs() const override { return 1; }
    int getAxis() const { return axis; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
    vector<DataType> inferDataType(const TensorVec &inputs) const override;
};

} // namespace infini

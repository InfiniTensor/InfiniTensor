#pragma once
#include "core/operator.h"

namespace infini {
/**
 * @brief The linear dequantization operator.
 * It consumes a quantized tensor, a scale, and a zero point to compute
 * the full precision tensor.
 */
class DequantizeLinearObj : public OperatorObj {
    int axis;

  public:
    /**
     * @brief Construct a new DequantizeLinear object.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param input The input tensor.
     * @param scale Scale for input.
     * @param zero_point Zero point for input.
     * @param outputs The output tensors.
     * @param axis The axis of the dequantizing dimension of the input tensor.
     */
    DequantizeLinearObj(GraphObj *graph, Tensor input, Tensor scale,
                        Tensor zero_pointr, Tensor output, int axis);
    OP_CLONE(DequantizeLinearObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;

    std::string toString() const override;
    int numInputs() const override { return inputs.size(); }
    int numOutputs() const override { return 1; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;

    vector<DataType> inferDataType(const TensorVec &inputs) const override;
};

} // namespace infini

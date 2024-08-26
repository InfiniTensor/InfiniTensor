#pragma once
#include "core/operator.h"

namespace infini {
/**
 * @brief Add data at the out side of a tensor.
 *
 */
class AscendDequantObj : public OperatorObj {
  private:
    vector<float> scale, offset;
    bool sqrtMode = false;

  public:
    // pad for appointed axises,if axis is empty,then pad for all axises.
    /**
     * @brief Construct a new Pad object.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param input The input tensor.
     * @param output The padded tensor.
     * @param scale scale value in quantization.目前不支持broadcast。
     * @param offset offset value in quantization.目前不支持broadcast。
     * @param sqrtMode Specify the logic for scale to participate in the
     * calculation. When the value is true, scale=scale*scale.
     */
    AscendDequantObj(GraphObj *graph, Tensor input, Tensor output,
                     const vector<float> &scale, const vector<float> &offset,
                     bool sqrtMode = false);
    OP_CLONE(AscendDequantObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;
    std::string toString() const override;
    int numInputs() const override { return inputs.size(); }
    int numOutputs() const override { return 1; }
    vector<float> getscale() const { return scale; }
    vector<float> getoffset() const { return offset; }
    bool getsqrtMode() const { return sqrtMode; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;

    vector<DataType> inferDataType(const TensorVec &inputs) const override;
};
} // namespace infini

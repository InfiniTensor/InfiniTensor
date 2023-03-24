#pragma once
#include "core/operator.h"

namespace infini {
/**
 * @brief Copy a tensor along a centain dimension for multiple times.
 */
class DropoutObj : public OperatorObj {
    float ratio;
    // bool training_mode; // TODO must be false.

  public:
    /**
     * @brief Dropout takes an input floating-point tensor, an input ratio
     * (floating-point scalar) and an input training_mode (boolean scalar). It
     * produces two tensor outputs, output (floating-point tensor) and mask
     * (bool tensor). If training_mode is true then the output Y will be a
     * random dropout; Note that this Dropout scales the masked input data by
     * the following equation, so to convert the trained model into inference
     * mode, the user can simply not pass training_mode input or set it to
     * false.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param data The input tensor.
     * @param output The output tensor.
     * @param mask The mask tensor.
     * @param ratio The ratio of random dropout, with value in [0, 1). If this
     * input was not set, or if it was set to 0, the output would be a simple
     * copy of the input. If it’s non-zero, output will be a random dropout of
     * the scaled input, which is typically the case during training.
     * @param training_mode  If set to true then it indicates dropout is being
     * used for training. It is an optional value hence unless specified
     * explicitly, it is false. If it is false, ratio is ignored and the
     * operation mimics inference mode where nothing will be dropped from the
     * input data and if mask is requested as output it will contain all ones.
     */
    DropoutObj(GraphObj *graph, Tensor data, Tensor output, Tensor mask,
               float ratio, bool training_mode);
    OP_CLONE(DropoutObj);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 2; }
    float getRatio() const { return ratio; }
    bool getTrainingMode() const { return false; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};
} // namespace infini

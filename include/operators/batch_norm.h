#pragma once
#include "core/operator.h"

namespace infini {
/**
 * @brief See https://arxiv.org/abs/1502.03167 for the detail of batch
 * normalization.
 *
 */
class BatchNormObj : public OperatorObj {
    float momentum, eps;
    bool trainingMode;

  public:
    /**
     * @brief Construct a new BatchNorm object.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param input The input tensor of BatchNorm. For image data, the input
     * shape is usually [N, C, H, W].
     * @param output The output tensor of BatchNorm, which should have the same
     * shape as the input tensor.
     * @param mean The mean tensor, which has a shape of [C].
     * @param var The var tensor, which has a shape of [C].
     * @param scale The scale tensor, which has a shape of [C].
     * @param bias The bias tensor, which has a shape of [C].
     * @param momentum Factor used in computing the running mean and variance.
     * Default is 0.9.
     * @param eps The epsilon value to use to avoid division by zero. Default is
     * 1e-5.
     * @param trainingMode Set to true when used for training.
     */
    BatchNormObj(GraphObj *graph, Tensor input, Tensor output, Tensor mean,
                 Tensor var, Tensor scale, Tensor bias, float momentum = 0.9,
                 float eps = 1e-5, bool trainingMode = false);
    OP_CLONE(BatchNormObj);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;
    std::string toString() const override;

    // output size will be 3 when training
    int numInputs() const override { return 5; }
    int numOutputs() const override { return outputs.size(); }
    float getMomentum() const { return momentum; }
    float getEps() const { return eps; }
    bool getTrainingMode() const { return trainingMode; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;

    vector<DataType> inferDataType(const TensorVec &inputs) const override;
};
} // namespace infini

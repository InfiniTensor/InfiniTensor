#pragma once
#include "core/operator.h"

namespace infini {
/**
 * @brief Fused RMSNorm Operator
 *
 */
class RMSNormObj : public OperatorObj {
    int dim = 0;
    float epsilon;

  public:
    /**
     * @brief Construct a new RMSNorm object.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param input The input tensor.
     * @param output The output tensor.
     */
    RMSNormObj(GraphObj *graph, Tensor input, Tensor weight, Tensor output,
               float epsilon = 1e-6f);
    OP_CLONE(RMSNormObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;

    std::string toString() const override;
    int numInputs() const override { return 2; }
    int numOutputs() const override { return 1; }
    int getDim() const { return dim; }
    float getEpsilon() const { return epsilon; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};
} // namespace infini

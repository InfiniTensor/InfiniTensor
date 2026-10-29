#pragma once
#include "core/operator.h"

namespace infini {
/**
 * @brief Fused RMSNorm Operator
 *
 */
class RMSNormObj : public OperatorObj {
    int dim;

  public:
    /**
     * @brief Construct a new RMSNorm object.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param input The input tensor.
     * @param output The output tensor.
     */
    RMSNormObj(GraphObj *graph, Tensor input, Tensor weight, Tensor output);
    OP_CLONE(RMSNormObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;

    std::string toString() const override;
    int numInputs() const override { return 2; }
    int numOutputs() const override { return 1; }
    int getDim() const { return dim; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};
} // namespace infini

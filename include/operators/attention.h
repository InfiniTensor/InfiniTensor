#pragma once
#include "core/operator.h"

namespace infini {
/**
 * @brief Return elements, either from X or Y, depending on condition.
 *
 */
class AttentionObj : public OperatorObj {

  public:
    /**
     * @brief Construct a new Attention object.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param inputX The input tensor Q.
     * @param inputY The input tensor K.
     * @param output The output tensor.
     * @param inputV The input tensor V.
     */
    AttentionObj(GraphObj *graph, Tensor inputQ, Tensor inputK, Tensor inputV,
             Tensor output);
    OP_CLONE(AttentionObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    std::string toString() const override;
    int numInputs() const override { return inputs.size(); }
    int numOutputs() const override { return 1; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

} // namespace infini

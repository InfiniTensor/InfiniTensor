#pragma once
#include "core/operator.h"

namespace infini {
/**
 * @brief Return elements, either from X or Y, depending on condition.
 *
 */
class WhereObj : public OperatorObj {

  public:
    /**
     * @brief Construct a new Where object.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param inputX The input tensor X.
     * @param inputY The input tensor Y.
     * @param output The output tensor.
     * @param condition The condition tensor.
     */
    WhereObj(GraphObj *graph, Tensor inputX, Tensor inputY, Tensor condition,
             Tensor output);
    OP_CLONE(WhereObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;

    std::string toString() const override;
    int numInputs() const override { return inputs.size(); }
    int numOutputs() const override { return 1; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

} // namespace infini

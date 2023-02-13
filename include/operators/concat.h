#pragma once
#include "core/operator.h"

namespace infini {
/**
 * @brief Concatenate several tensors into one. All the input tensors should
 * have the same shape except for the concatenated dimension.
 *
 */
class ConcatObj : public OperatorObj {
    int dim;

  public:
    /**
     * @brief Construct a new Concat object.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param inputs The input tensors to be concatenated.
     * @param output Concatenated tensor.
     * @param dim The dimension to concatenate on.
     */
    ConcatObj(GraphObj *graph, TensorVec inputs, Tensor output, int dim);
    OP_CLONE(ConcatObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    std::string toString() const override;
    int numInputs() const override { return inputs.size(); }
    int numOutputs() const override { return 1; }
    int getDim() const { return dim; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};
} // namespace infini

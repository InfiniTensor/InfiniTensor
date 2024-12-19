#pragma once
#include "core/operator.h"

namespace infini {
/**
 * @brief Return elements, either from X or Y, depending on condition.
 *
 */
class ScatterNDObj : public OperatorObj {

  public:
    /**
     * @brief Construct a new ScatterND object.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param data The data tensor.
     * @param indices The index tensor
     * @param updates The update tensor
     * @param output The output tensor.
     */
    ScatterNDObj(GraphObj *graph, Tensor data, Tensor indices, Tensor updates,
                 Tensor output);
    OP_CLONE(ScatterNDObj);

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
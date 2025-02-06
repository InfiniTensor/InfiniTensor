#pragma once
#include "core/operator.h"

namespace infini {
/**
 * @brief Return elements, either from X or Y, depending on condition.
 *
 */
class ScatterElementsObj : public OperatorObj {
    int axis;
    std::string reduction;

  public:
    /**
     * @brief Construct a new ScatterElements object.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param data The data tensor.
     * @param indices The index tensor
     * @param updates The update tensor
     * @param output The output tensor.
     */
    ScatterElementsObj(GraphObj *graph, Tensor data, Tensor indices,
                       Tensor updates, Tensor output, int axis = 0,
                       std::string reduction = "none");
    OP_CLONE(ScatterElementsObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;

    std::string toString() const override;
    int numInputs() const override { return inputs.size(); }
    int numOutputs() const override { return 1; }
    int getAxis() const { return axis; }
    std::string getReduction() const { return reduction; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
    vector<DataType> inferDataType(const TensorVec &inputs) const override;
};

} // namespace infini

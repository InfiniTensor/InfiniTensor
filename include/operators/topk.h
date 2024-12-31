#pragma once
#include "core/operator.h"

namespace infini {
/**
 * @brief Return elements, either from X or Y, depending on condition.
 *
 */
class TopKObj : public OperatorObj {
    Shape K;
    int axis;
    int Largest;
    int sorted;

  public:
    /**
     * @brief Construct a new TopK object.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param input The input tensor.
     * @param K topk.
     * @param outputs The output tensor=(Indices, Values).
     */
    TopKObj(GraphObj *graph, Tensor input, std::optional<TensorVec> outputs,
            Shape K, int axis = -1, int Largest = 1, int sorted = 1);
    OP_CLONE(TopKObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;

    std::string toString() const override;
    int numInputs() const override { return 1; }
    inline Shape getTopk() const { return K; }
    int numOutputs() const override { return 2; }
    int getAxis() const { return axis; }
    int getLargest() const { return Largest; }
    int getSorted() const { return sorted; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
    vector<DataType> inferDataType(const TensorVec &inputs) const override;
};

} // namespace infini

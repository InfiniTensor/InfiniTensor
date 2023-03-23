#pragma once
#include "core/operator.h"

namespace infini {
/**
 * @brief Split a tensor into multiple ones.
 *
 */
class SplitObj : public OperatorObj {
    int dim, num;      // split dim;Average split num or outputs size
    vector<int> ratio; // output dim ratio
  public:
    /**
     * @brief Construct a new Split object.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param input The input tensor.
     * @param outputs The output tensors after splitting.
     * @param dim The dimension to split.
     * @param num The number of output tensors. The input tensor is split into
     * `num` evenly chunk along dimension `dim. The last chunk will be smaller
     * if the input tensor cannot be evenly split.
     */
    SplitObj(GraphObj *graph, Tensor input, std::optional<TensorVec> outputs,
             int dim, int num);
    /**
     * @brief Construct a new Split object.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param input The input tensor.
     * @param outputs The output tensors after splitting.
     * @param dim The dimension to split.
     * @param ratio The size of dimension `dim` for the output tensors after
     * splitting.
     */
    SplitObj(GraphObj *graph, Tensor input, std::optional<TensorVec> outputs,
             int dim, const vector<int> &ratio);
    OP_CLONE(SplitObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return num; }
    int getDim() const { return dim; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};
} // namespace infini

#pragma once
#include "core/operator.h"

namespace infini {
/**
 * @brief Copy a tensor along a centain dimension for multiple times.
 *
 */
class ExtendObj : public OperatorObj {
    int dim, num; // copy num times at the dim.

  public:
    /**
     * @brief Construct a new Extend object.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param input The input tensor.
     * @param output The extened tensor.
     * @param dim The dimension to extend on.
     * @param num The number of times to copy when extending. The dimension size
     * of `dim` becomes `num+1` times after extending.
     */
    ExtendObj(GraphObj *graph, Tensor input, Tensor output, int dim,
              int num = 1);
    OP_CLONE(ExtendObj);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }
    int getDim() const { return dim; }
    int getNum() const { return num; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};
} // namespace infini

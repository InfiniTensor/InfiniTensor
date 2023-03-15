#pragma once
#include "core/operator.h"

namespace infini {
/**
 * @brief Compute the mean of input tensor's elements along certain axes.
 *
 */
class ReduceMeanObj : public OperatorObj {
    set<int> axes; // axis to reduce
    bool keepDims;

  public:
    /**
     * @brief Construct a new ReduceMean object.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param input The input tensor.
     * @param output The output tensor.
     * @param axes Axes to reduce.
     * @param keepDims Keep the reduced dimensions or not.
     */
    ReduceMeanObj(GraphObj *graph, Tensor input, Tensor output,
                  const optional<vector<int>> &axes, bool keepDims = true);
    OP_CLONE(ReduceMeanObj);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }

    bool isReduced(int idx) const;
    const set<int> &getAxes() const { return axes; }
    bool getKeepDims() const { return keepDims; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

} // namespace infini

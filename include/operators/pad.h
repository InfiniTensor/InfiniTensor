#pragma once
#include "core/operator.h"

namespace infini {
/**
 * @brief Add data at the out side of a tensor.
 *
 */
class PadObj : public OperatorObj {
    // the number of start and end pad values for all dims.
    vector<int> pads;

  public:
    // pad for appointed axises,if axis is empty,then pad for all axises.
    /**
     * @brief Construct a new Pad object.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param input The input tensor.
     * @param output The padded tensor.
     * @param pads Add padding elements at the begining and end of each axis.
     * Suppose that padding axes are [x1, x2, ...], then pads's format is
     * [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
     * @param axes Pad for appointed axes. If axis is empty, pad for all axes.
     */
    PadObj(GraphObj *graph, Tensor input, Tensor output,
           const vector<int> &pads, const optional<vector<int>> &axes);
    OP_CLONE(PadObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;
    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }
    Shape getPads() const { return pads; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};
} // namespace infini

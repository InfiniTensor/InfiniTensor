#pragma once
#include "core/operator.h"
#include "utils/operator_utils.h"

namespace infini {
/*
 *
 * @brief The Argmax operator returns the indices of the maximum values along a
 *
 */
class ArgMaxObj : public OperatorObj {
    int axis;
    int keepDims;
    int selectLastIndex;

  public:
    ArgMaxObj(GraphObj *graph, Tensor input, Tensor output, int axis = 0,
              int keepDims = 1, int selectLastIndex = 0);
    OP_CLONE(ArgMaxObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;
    vector<DataType> inferDataType(const TensorVec &inputs) const override;

    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }

    int getAxis() const { return get_real_axis(axis, inputs[0]->getRank()); }
    int getKeepDims() const { return keepDims; }
    int getSelectLastIndex() const { return selectLastIndex; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};
} // namespace infini
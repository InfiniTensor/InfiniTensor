#pragma once
#include "core/operator.h"

namespace infini {
  /*
   *
   * @brief The Argmax operator returns the indices of the maximum values along a
   * 
   */
class ArgMaxObj : public OperatorObj {
    int axis;
    bool keepDims;
    bool selectLastIndex;

  public:
    ArgMaxObj(GraphObj *graph, Tensor input, Tensor output, int axis = 0,
              bool keepDims = true, bool selectLastIndex = false);
    OP_CLONE(ArgMaxObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;

    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }
    
    int getAxis() const { return axis; }
    bool getKeepDims() const { return keepDims; }
    bool getSelectLastIndex() const { return selectLastIndex; }
    Tensor getIndicesTensor() const { return outputs[0]; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};
} // namespace infini
#pragma once
#include "core/operator.h"

namespace infini {
class ReduceMeanObj : public OperatorObj {
    set<int> axis; // axis to reduce
    bool keepDims;

  public:
    ReduceMeanObj(GraphObj *graph, Tensor input, Tensor output,
                  const optional<const vector<int>> &axis,
                  bool keepDims = true);
    OP_CLONE(ReduceMeanObj);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }

    bool isReduced(int idx) const;
    bool getKeepDims() const { return keepDims; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

} // namespace infini

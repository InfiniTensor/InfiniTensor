#pragma once
#include "core/operator.h"

namespace infini {

class AnyObj : public OperatorObj {
  private:
    string kernelName;
    vector<int> attr;

  public:
    AnyObj(GraphObj *graph, const TensorVec &inputs, const TensorVec &outputs,
           const string &kernelName, const vector<int> &attr);

    OP_CLONE(AnyObj);

    string toString() const override;

    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    int numInputs() const override { return inputs.size(); }
    int numOutputs() const override { return outputs.size(); }

    const string getKernelName() const;
    void setAttr(int i, int v) { attr[i] = v; }
    vector<int> getOpAttrVector() const override;
    vector<int> getWorkloadVector() const override;
};

} // namespace infini

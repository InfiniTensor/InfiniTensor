#pragma once
#include "core/tensor.h"

namespace it {

class OperatorNode : public Object {
  protected:
    // OpType type;
    TensorVec inputs;
    TensorVec outputs;
    // vector<WRef<Operator>> predecessors;
    // vector<WRef<Operator>> successors;
  public:
    OperatorNode(TensorVec inputs, TensorVec outputs)
        : inputs(inputs), outputs(outputs) {}
    string toString() const override;
    // Operator(TensorVec inputs) : inputs(inputs) {}

    virtual ~OperatorNode() {}
};
} // namespace it
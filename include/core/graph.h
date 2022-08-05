#pragma once
#include "core/operator.h"
#include "core/tensor.h"

namespace it {

class GraphNode : public Object {
  protected:
    TensorVec tensors;
    TensorVec inputs;
    TensorVec outputs;
    OpVec ops;

  public:
    // Graph(OpVec oplist);
    string toString() const override;

    void addOp(Operator op) { ops.push_back(op); };
    const TensorVec &getTensors() const { return tensors; }
    const TensorVec &getInputs() const { return inputs; }
    const TensorVec &getOutputs() const { return outputs; }
    const OpVec &getOperators() const { return ops; }
    // TensorVec &getInputs();
    // TensorVec &getOutputs();

    Tensor addTensor(Shape dim) {
        Tensor tensor = make_ref<TensorNode>(dim);
        tensors.emplace_back(tensor);
        return tensor;
    }

    void updateConnection();

    // TODO
    // bool compute();

    // TODO: move to another class
    // bool exportOnnx(const char *path);
    // bool importOnnx(const char *net);
};

} // namespace it
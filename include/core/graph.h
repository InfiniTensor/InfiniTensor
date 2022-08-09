#pragma once
#include "core/operator.h"
#include "core/tensor.h"

namespace infini {

// TODO: graph should be attached to a context
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

    Tensor addTensor(Shape dim, DataType dtype = DataType::Int32) {
        Tensor tensor = make_ref<TensorNode>(dim, dtype);
        tensors.emplace_back(tensor);
        return tensor;
    }

    void dataMalloc();

  private:
    // TODO: updateConnection
    /**
     * @brief Add reverse connections and Op relationship in ctor.
     */
    void updateConnection();

    // TODO: move to another class
    // bool exportOnnx(const char *path);
    // bool importOnnx(const char *net);
};

} // namespace infini

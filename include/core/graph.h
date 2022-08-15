#pragma once
#include "core/operator.h"
#include "core/tensor.h"

namespace infini {

// TODO: graph should be attached to a context
class GraphObj : public Object {
  protected:
    TensorVec tensors;
    TensorVec inputs;
    TensorVec outputs;
    OpVec ops;

  public:
    // Graph(OpVec oplist);
    string toString() const override;

    Tensor addTensor(Shape dim, DataType dtype = DataType::Int32);

    /**
     * @brief Add an operator and create its outputs. Output tensor arguments
     * should be empty Refs (e.g., nullptr).
     */
    template <typename T, typename... Args> Ref<T> addOp(Args &&...args) {
        Ref<T> op = make_ref<T>(this, std::forward<Args>(args)...);
        ops.push_back(op);
        return op;
    }

    /**
     * @brief Add an operator with its outputs specified.
     */
    template <typename T, typename... Args>
    Ref<T> addOpWithOutputs(Args &&...args) {
        Ref<T> op = make_ref<T>(nullptr, std::forward<Args>(args)...);
        ops.push_back(op);
        return op;
    }

    const TensorVec &getTensors() const { return tensors; }
    const TensorVec &getInputs() const { return inputs; }
    const TensorVec &getOutputs() const { return outputs; }
    const OpVec &getOperators() const { return ops; }
    // TensorVec &getInputs();
    // TensorVec &getOutputs();

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

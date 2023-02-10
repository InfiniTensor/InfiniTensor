#pragma once
#include "core/operator.h"
#include "core/tensor.h"

namespace infini {

class GraphObj : public Object {
  protected:
    Runtime runtime;
    TensorVec tensors;
    // TODO: whether to record input and output tensors
    // TensorVec inputs;
    // TensorVec outputs;
    OpVec ops;

  public:
    GraphObj(Runtime runtime) : runtime(runtime){};
    GraphObj(Runtime runtime, OpVec ops_in);
    string toString() const override;
    Runtime getRuntime() const { return runtime; }

    Tensor addTensor(Shape dim, DataType dtype = DataType::Float32);
    Tensor addTensor(const Tensor &tensor);
    TensorVec addTensor(const TensorVec &tensors);
    Tensor cloneTensor(const Tensor &tensor) {
        auto ret = addTensor(tensor->clone(runtime));
        return ret;
    }

    /**
     * @brief Add an operator and create its outputs. Output tensor arguments
     * should be empty Refs (e.g., nullptr).
     */
    template <typename T, typename... Args> Ref<T> addOp(Args &&...args) {
        Ref<T> op = infini::make_ref<T>(this, std::forward<Args>(args)...);
        addOperatorAndConnect(op);
        return op;
    }

    /**
     * @brief Add an operator with its outputs specified.
     */
    template <typename T, typename... Args>
    Ref<T> addOpWithOutputs(Args &&...args) {
        Ref<T> op = infini::make_ref<T>(nullptr, std::forward<Args>(args)...);
        addOperatorAndConnect(op);
        return op;
    }

    const TensorVec &getTensors() const { return tensors; }
    const TensorVec getInputsAndWeights() const {
        TensorVec ret;
        for (auto t : tensors)
            if (!t->getOutputOf())
                ret.emplace_back(t);
        return ret;
    }
    const TensorVec getOutputs() const {
        TensorVec ret;
        for (auto t : tensors)
            if (t->getInputOf().empty())
                ret.emplace_back(t);
        return ret;
    }
    const OpVec &getOperators() const { return ops; }
    OpVec getComputeOps() const;

    void dataMalloc();

  private:
    /**
     * @brief Add reverse connections and Op relationship in ctor.
     */
    void addOperatorAndConnect(const Operator &op);

    // TODO: move to another class
    // bool exportOnnx(const char *path);
    // bool importOnnx(const char *net);
};

} // namespace infini

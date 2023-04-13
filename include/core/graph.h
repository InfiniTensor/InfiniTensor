#pragma once
#include "core/operator.h"
#include "core/tensor.h"

namespace infini {

class GraphObj : public Object {
  protected:
    Runtime runtime;
    TensorVec tensors;
    OpVec ops;

  public:
    explicit GraphObj(Runtime runtime) : runtime(runtime), sorted(false){};
    GraphObj(Runtime runtime, OpVec ops_in);
    string toString() const override;
    Runtime getRuntime() const { return runtime; }

    Tensor addTensor(Shape dim, DataType dtype = DataType::Float32);
    Tensor addTensor(const Tensor &tensor);
    TensorVec addTensor(const TensorVec &tensors);
    /**
     * @brief Clone a tensor and add it to the graph.
     */
    Tensor cloneTensor(const Tensor &tensor) {
        return addTensor(tensor->clone(runtime));
    }

    const TensorVec &getTensors() const { return tensors; }
    const OpVec &getOperators() const { return ops; }
    OpVec getComputeOps() const;

    /**
     * Sort the nodes in topological order.
     * It returns true if the sorting is successful.
     * Otherwise false is returned, means that there are rings in the graph,
     * so the topological sorting fails.
     */
    bool topo_sort();

    void dataMalloc();

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

    /**
     * @brief Gets input tensors of this graph.
     */
    inline TensorVec getInputs() const {
        TensorVec ret;
        for (const auto &t : tensors)
            if (!t->getSource())
                ret.emplace_back(t);
        return ret;
    }

    /**
     * @brief Gets output tensors of this graph.
     */
    inline TensorVec getOutputs() const {
        TensorVec ret;
        for (const auto &t : tensors)
            if (t->getTargets().empty())
                ret.emplace_back(t);
        return ret;
    }

    bool selfCheck() const;

  private:
    /**
     * @brief Add reverse connections and Op relationship in ctor.
     */
    void addOperatorAndConnect(const Operator &op);

    /**
     * @brief If the nodes is sorted in topological order.
     */
    bool sorted;
};

} // namespace infini

#pragma once

#include "core/op_type.h"
#include "core/tensor.h"
#include "utils/infiniop_utils.h"

namespace infini {
using KernelAttrs = std::tuple<Device, OpType::underlying_t>;

struct OpPerfKey {
    HashType hash;
    OpType::underlying_t opType;
    vector<int> attrs;

  public:
    // FIXME: default ctor should be deleted but json requires it. Solution:
    // https://github.com/nlohmann/json#how-can-i-use-get-for-non-default-constructiblenon-copyable-types
    OpPerfKey() = default;
    OpPerfKey(HashType hash, OpType opType, vector<int> attrs = {})
        : hash(hash), opType(opType.underlying()), attrs(attrs) {}
    bool operator==(const OpPerfKey &rhs) const {
        if (hash != rhs.hash)
            return false;
        if (opType != rhs.opType)
            return false;
        if (attrs != rhs.attrs)
            return false;
        return true;
    }

    // TODO: remove this function after we use unordered_map in PerfEngine
    bool operator<(const OpPerfKey &rhs) const {
        if (hash != rhs.hash)
            return hash < rhs.hash;
        if (opType != rhs.opType)
            return opType < rhs.opType;
        if (attrs.size() != rhs.attrs.size())
            return attrs.size() < rhs.attrs.size();
        for (size_t i = 0; i < attrs.size(); ++i)
            if (attrs[i] != rhs.attrs[i])
                return attrs[i] < rhs.attrs[i];
        return false;
    }
};

class GraphObj;
class OperatorObj : public Object {
    friend class GraphObj;

  protected:
    OpType type;
    TensorVec inputs;
    TensorVec outputs;
    vector<WRef<OperatorObj>> predecessors;
    vector<WRef<OperatorObj>> successors;
    void *opDesc;

  public:
    OperatorObj(OpType opType, TensorVec inputs, TensorVec outputs);
    virtual optional<vector<Shape>> inferShape(const TensorVec &inputs) = 0;
    virtual vector<DataType> inferDataType(const TensorVec &inputs) const;
    /**
     * @brief Constructs outputs (if requried) and check whether the operator is
     * valid.
     *
     * @param graph If graph is not nullptr, outputs should be created in this
     * function.
     */
    bool checkValid(GraphObj *graph);
    virtual void initInfiniOp(const Runtime context);
    OpPerfKey getOpPerfKey() const;
    /**
     * @brief Hash operator attributes. Input and output shapes are not
     * considered.
     */
    HashType hash() const;

  public:
  public: // getter and setter
    const TensorVec &getInputs() const { return inputs; }
    const TensorVec &getOutputs() const { return outputs; }
    void *getOpDesc() const { return opDesc; }
    Tensor getInputs(size_t i) const { return inputs.at(i); }
    Tensor getOutput() const {
        IT_ASSERT(outputs.size() == 1, "Unimplemented");
        return outputs[0];
    }
    Tensor getOutput(size_t i) const {
        IT_ASSERT(i < outputs.size(), "Index exceeded");
        return outputs.at(i);
    }
    OpVec getPredecessors() const { return wrefs_to_refs(predecessors); }
    OpVec getSuccessors() const { return wrefs_to_refs(successors); }
    OpType getOpType() const { return type; }
    // HACK: set correct data type
    DataType getDType() const { return getInputs(0)->getDType(); }
    DataType getOutDType() const { return getOutput()->getDType(); }
    virtual int numInputs() const = 0;
    virtual int numOutputs() const = 0;

    /**
     * @brief Clone this operator and replace its inputs and outputs.
     *
     * @param newInputs
     * @param newOutputs
     * @return Operator
     */
    virtual Operator clone(const TensorVec &newInputs,
                           const TensorVec &newOutputs) const = 0;

  protected:
    optional<vector<Shape>> inferShape();
    vector<DataType> inferDataType() const;

  private:
    /**
     * @brief The returned vector includes operator attributes, such as paddings
     * in Conv and transpose in Matmul. However, the input and output shapes are
     * not taken into consideration.
     */
    virtual vector<int> getOpAttrVector() const { IT_TODO_HALT(); }
    /**
     * @brief Besides operator attributes, the returned vector includes input
     * and output shapes.
     */
    virtual vector<int> getWorkloadVector() const { IT_TODO_HALT(); }

    void addPredecessors(const Operator &op) { predecessors.emplace_back(op); }
    void addSuccessors(const Operator &op) { successors.emplace_back(op); }
    void removePredecessors(const Operator &op);
    void removeSuccessors(const Operator &op);
    void replaceInput(Tensor t1, Tensor t2);
};

#define OP_CLONE(OpObj)                                                        \
    virtual Operator clone(const TensorVec &newInputs,                         \
                           const TensorVec &newOutputs) const override {       \
        auto op = infini::make_ref<OpObj>(*this);                              \
        op->inputs = newInputs;                                                \
        op->outputs = newOutputs;                                              \
        op->predecessors.clear();                                              \
        op->successors.clear();                                                \
        IT_ASSERT(op->checkValid(nullptr));                                    \
        return op;                                                             \
    }

} // namespace infini

namespace std {
template <> struct hash<infini::OpPerfKey> {
    size_t operator()(const infini::OpPerfKey &key) const { return key.hash; }
};
} // namespace std

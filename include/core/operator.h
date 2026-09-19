#pragma once

#include "core/op_type.h"
#include "core/tensor.h"

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

/// @brief One dimension of one input that an output dimension follows.
struct DimSource {
    size_t input; ///< Index into the operator's inputs.
    size_t dim;   ///< Which dimension of that input.
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

  public:
    OperatorObj(OpType opType, TensorVec inputs, TensorVec outputs);
    virtual optional<vector<Shape>> inferShape(const TensorVec &inputs) = 0;
    virtual vector<DataType> inferDataType(const TensorVec &inputs) const;
    /**
     * @brief Works out the contents of the outputs, when they describe shapes
     * and follow from the shapes of the inputs. See `TensorObj::getShapeValue`.
     *
     * Called once the outputs exist, and again whenever shapes change. An
     * operator that cannot work its outputs out leaves them cleared, so that a
     * result computed for earlier shapes is never mistaken for a current one.
     * Operators that do not take part in shape computation need not override
     * this.
     */
    virtual void inferShapeValue() {}
    /**
     * @brief Which input dimensions dimension `dim` of output `output` was
     * worked out from.
     *
     * A computed dimension can change exactly when one of the dimensions it
     * follows can, so this is what says whether it is settled. Returning an
     * empty list claims the dimension follows nothing a caller may vary, and so
     * is settled whatever the graph is given -- an output channel count, say,
     * which comes from a weight.
     *
     * The default names every dimension of every input, which is the
     * conservative reading: it settles an output only when the whole of every
     * input is settled. An operator that knows better says so, and one that
     * does not stays correct.
     *
     * Only shapes are being described, so the answer must not depend on the
     * sizes those dimensions currently hold.
     */
    virtual vector<DimSource> dimSources(size_t output, size_t dim) const;
    /**
     * @brief Constructs outputs (if requried) and check whether the operator is
     * valid.
     *
     * @param graph If graph is not nullptr, outputs should be created in this
     * function.
     */
    bool checkValid(GraphObj *graph);
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

    /// @brief Clears the shape values of the outputs and reports whether this
    /// operator can work out new ones.
    ///
    /// An operator can only pass shapes along once every input carries a known
    /// shape value and every output is able to hold one. Clearing first means a
    /// value left over from an earlier shape never survives as a stale answer,
    /// whether this operator goes on to write a new one or gives up.
    bool beginShapeValueUpdate();

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

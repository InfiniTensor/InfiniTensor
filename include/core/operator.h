#pragma once
#include "core/tensor.h"
namespace infini {

enum class OpType {
    Unknown = 0,
    // linear
    Conv = 100,
    ConvBackwardFilter,
    ConvBackwardData,
    Matmul,
    ConvTrans,
    ConvTransNHWC,
    ConvNHWC,
    G2BMM,
    GBMM,
    Pad,
    Slice,
    Concat,
    Split,
    Transpose,
    Extend,
    MaxPool,
    AvgPool,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Gather,
    ReduceMean,
    Reshape,
    Flatten,
    Identity,
    // element wise
    BatchNorm = 200,
    Softmax,
    Activation,
    Relu,
    ReluBackward,
    PRelu,
    Sigmoid,
    SigmoidBackward,
    Tanh,
    TanhBackward,
    Abs,
    Sin,
    Cos,
    Tan,
    ASin,
    ACos,
    ATan,
    SinH,
    CosH,
    TanH,
    ASinH,
    ACosH,
    ATanH,
    Resize,
    Arange,
    Shape,
    Copy,
    Ceil,
    Floor,
    Clip,
    Erf,
    Exp,
    Fill,
    Log,
    L2Loss,
    Maximum,
    Minimum,
    MSELoss,
    Neg,
    Power,
    Reciprocal,
    Sqrt,
    Rsqrt,
    Cast,
    FloorDiv,
    FloorMod,
    Det,
    Round,
    Square,
    SquaredDifference,
    Hardtanh,
    Equal,
    NotEqual,
    GreaterThan,
    GreaterEqual,
    LessThan,
    LessEqual,
    And,
    Or,
    Xor,
    Not,
    BitAnd,
    BitOr,
    BitXor,
    BitNot,
    BitLeftShift,
    BitRightShift,
    Dropout,
    //
    MemBound = 300,
    //
    Conv2dReduce = 400,
    Conv2dReduceTranspose,
    Any
};

using KernelAttrs = std::tuple<Device, OpType, DataType>;

class OpRegistry {
  public:
    static std::string getOpName(OpType opType) {
#define FOP(op)                                                                \
    case OpType::op:                                                           \
        return #op

        switch (opType) {
            FOP(Unknown);
            // linear
            FOP(Conv);
            FOP(ConvBackwardFilter);
            FOP(ConvBackwardData);
            FOP(Matmul);
            FOP(ConvTrans);
            FOP(ConvTransNHWC);
            FOP(ConvNHWC);
            FOP(G2BMM);
            FOP(GBMM);
            FOP(Pad);
            FOP(Slice);
            FOP(Concat);
            FOP(Split);
            FOP(Transpose);
            FOP(Extend);
            FOP(MaxPool);
            FOP(AvgPool);
            FOP(Add);
            FOP(Sub);
            FOP(Mul);
            FOP(Div);
            FOP(Pow);
            FOP(Gather);
            FOP(ReduceMean);
            FOP(Reshape);
            FOP(Identity);
            FOP(Shape);
            FOP(Flatten);
            // element wise
            FOP(BatchNorm);
            FOP(Softmax);
            FOP(Activation);
            FOP(Relu);
            FOP(ReluBackward);
            FOP(PRelu);
            FOP(Sigmoid);
            FOP(SigmoidBackward);
            FOP(Tanh);
            FOP(TanhBackward);
            FOP(Abs);
            FOP(Sin);
            FOP(Cos);
            FOP(Tan);
            FOP(ASin);
            FOP(ACos);
            FOP(ATan);
            FOP(SinH);
            FOP(CosH);
            FOP(TanH);
            FOP(ASinH);
            FOP(ACosH);
            FOP(ATanH);
            FOP(Copy);
            FOP(Ceil);
            FOP(Floor);
            FOP(Clip);
            FOP(Erf);
            FOP(Exp);
            FOP(Fill);
            FOP(Log);
            FOP(L2Loss);
            FOP(Maximum);
            FOP(Minimum);
            FOP(MSELoss);
            FOP(Neg);
            FOP(Power);
            FOP(Reciprocal);
            FOP(Sqrt);
            FOP(Rsqrt);
            FOP(Cast);
            FOP(FloorDiv);
            FOP(FloorMod);
            FOP(Det);
            FOP(Round);
            FOP(Square);
            FOP(SquaredDifference);
            FOP(Hardtanh);
            FOP(Equal);
            FOP(NotEqual);
            FOP(GreaterThan);
            FOP(GreaterEqual);
            FOP(LessThan);
            FOP(LessEqual);
            FOP(And);
            FOP(Or);
            FOP(Xor);
            FOP(Not);
            FOP(BitAnd);
            FOP(BitOr);
            FOP(BitXor);
            FOP(BitNot);
            FOP(BitLeftShift);
            FOP(BitRightShift);
            //
            FOP(MemBound);
            //
            FOP(Conv2dReduce);
            FOP(Conv2dReduceTranspose);
            FOP(Any);
        default:
            IT_ASSERT(false, "Unknown OpType " +
                                 std::to_string(enum_to_underlying(opType)));
            break;
        }
#undef FOP
    }
};

enum class ActType {
    None,
    Relu,
    Sigmoid,
    Tanh,
};

struct OpPerfKey {
    HashType hash;
    OpType opType;
    vector<int> attrs;

  public:
    // FIXME: default ctor should be deleted but json requires it. Solution:
    // https://github.com/nlohmann/json#how-can-i-use-get-for-non-default-constructiblenon-copyable-types
    OpPerfKey() = default;
    OpPerfKey(HashType hash, OpType opType, vector<int> attrs = {})
        : hash(hash), opType(opType), attrs(attrs) {}
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

  public:
    OperatorObj(OpType opType, TensorVec inputs, TensorVec outputs);
    virtual optional<vector<Shape>>
    inferShape(const TensorVec &inputs) const = 0;
    virtual vector<DataType> inferDataType(const TensorVec &inputs) const;
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

  public: // check Op type
    bool isLinearOp() const;
    bool isElementWiseOp() const;
    bool isSplitOp() const;
    bool isConcatOp() const;
    bool isComputeOp() const;
    bool isTransposeOp() const;
    bool isReshapeOp() const;
    bool isMemBoundOp() const;

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
    optional<vector<Shape>> inferShape() const;
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

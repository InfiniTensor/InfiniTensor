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
    DivDemo,
    DivNoNan,
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
    Copy,
    Ceil,
    Floor,
    Clip,
    Erf,
    Exp,
    Fill,
    Log_e,
    Log_2,
    Log_10,
    Log1p,
    L2Loss,
    Maximum,
    Minimum,
    MSELoss,
    NegTensor,
    Power,
    Reciprocal,
    Sqrt,
    Rsqrt,
    Transform,
    AddN,
    MulN,
    Cast,
    FloorDiv,
    FloorDivTrunc,
    FloorMod,
    FloorModTrunc,
    Cumsum,
    Cumprod,
    //
    MemBound = 300,
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
            FOP(DivDemo);
            FOP(DivNoNan);
            FOP(Pow);
            FOP(Gather);
            FOP(ReduceMean);
            FOP(Reshape);
            FOP(Identity);
            // element wise
            FOP(BatchNorm);
            FOP(Softmax);
            FOP(Activation);
            FOP(Relu);
            FOP(ReluBackward);
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
            FOP(Arange);
            FOP(Copy);
            FOP(Ceil);
            FOP(Floor);
            FOP(Clip);
            FOP(Erf);
            FOP(Exp);
            FOP(Fill);
            FOP(Log_e);
            FOP(Log_2);
            FOP(Log_10);
            FOP(Log1p);
            FOP(L2Loss);
            FOP(Maximum);
            FOP(Minimum);
            FOP(MSELoss);
            FOP(NegTensor);
            FOP(Power);
            FOP(Reciprocal);
            FOP(Sqrt);
            FOP(Rsqrt);
            FOP(Transform);
            FOP(AddN);
            FOP(MulN);
            FOP(Cast);
            FOP(FloorDiv);
            FOP(FloorDivTrunc);
            FOP(FloorMod);
            FOP(FloorModTrunc);
            FOP(Cumsum);
            FOP(Cumprod);
            //
            FOP(MemBound);
        default:
            IT_ASSERT(false);
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

class OperatorObj : public Object {
  protected:
    OpType type;
    TensorVec inputs;
    TensorVec outputs;
    vector<WRef<OperatorObj>> predecessors;
    vector<WRef<OperatorObj>> successors;

  public:
    OperatorObj(OpType opType, TensorVec inputs, TensorVec outputs);
    OperatorObj(OpType opType);
    void setInputs(TensorVec inputsTensor) { 
        inputs = inputsTensor;
        for (auto &t : inputs)
            IT_ASSERT(t != nullptr);
    }
    void setOutputs(TensorVec outputsTensor) { outputs = outputsTensor; }
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
    bool checkValid(GraphObj *graph, DataType type);
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
    void addPredecessors(const Operator &op) { predecessors.emplace_back(op); }
    void addSuccessors(const Operator &op) { successors.emplace_back(op); }
    OpVec getPredecessors() const { return wrefs_to_refs(predecessors); }
    OpVec getSuccessors() const { return wrefs_to_refs(successors); }
    OpType getOpType() const { return type; }
    // HACK: set correct data type
    DataType getDType() const { return getInputs(0)->getDType(); }
    virtual int numInputs() const = 0;
    virtual int numOutputs() const = 0;

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
};

} // namespace infini

namespace std {
template <> struct hash<infini::OpPerfKey> {
    size_t operator()(const infini::OpPerfKey &key) const { return key.hash; }
};
} // namespace std

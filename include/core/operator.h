#pragma once
#include "core/tensor.h"

namespace infini {

enum class OpType {
    Unknown = 0,
    // linear
    Conv = 100,
    Matmul,
    ConvTrans,
    G2BMM,
    GBMML,
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
    Identity,
    // element wise
    BatchNorm = 200,
    Softmax,
    Activation,
    Resize,
    //
    MemBound = 300,
};

enum class Device { CPU = 1, CUDA };

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
            FOP(Matmul);
            FOP(ConvTrans);
            FOP(G2BMM);
            FOP(GBMML);
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
            // element wise
            FOP(BatchNorm);
            FOP(Softmax);
            FOP(Activation);
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

struct OpAttrs {
  public:
    virtual bool operator<(const OpAttrs &rhs) const {
        IT_ASSERT(typeid(*this) == typeid(rhs), "OpAttrs type mismatch.");
        // Empty OpAttrs are equal
        return false;
    }
    virtual ~OpAttrs() {}
};

class OperatorNode : public Object {
  public:
  protected:
    OpType type;
    TensorVec inputs;
    TensorVec outputs;
    // vector<WRef<Operator>> predecessors;
    // vector<WRef<Operator>> successors;

  public:
    OperatorNode(OpType opType, TensorVec inputs, TensorVec outputs)
        : type(opType), inputs(inputs), outputs(outputs) {}
    virtual vector<Shape> computeShape() const = 0;
    virtual OpAttrs getOpAttrs() const = 0;

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
    // TensorVec getInputs() { return inputs; }
    const TensorVec &getInputs() const { return inputs; }
    // TensorVec getOutputs() { return outputs; }
    const TensorVec &getOutputs() const { return outputs; }
    Tensor getInputs(size_t i) { return inputs.at(i); }
    Tensor getOutput() const {
        IT_ASSERT(outputs.size() == 1, "Unimplemented");
        return outputs[0];
    }
    OpType getOpType() const { return type; }

    virtual int numInputs() const = 0;
    virtual int numOutputs() const = 0;
};

} // namespace infini
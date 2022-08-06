#pragma once
#include "core/tensor.h"

namespace it {

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

class OpRegistry {
  public:
    std::string getOpName(OpType opType) {
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

class MatmulNode : public OperatorNode {
  public:
    struct MatmulArgs {
        int b, m, n, k;
        // PET assume a row-major tensor layout. transA=false means default
        // dims, true means A should be transposed before matmul. This is in
        // oppsite to column-major BLAS.
        bool transA, transB;
        ActType act;
    };

  private:
    MatmulArgs args;

  public:
    MatmulNode(Tensor A, Tensor B, Tensor C, bool transA = false,
               bool transB = false, Tensor bias = nullptr,
               ActType act = ActType::None);

    std::string toString() const override;
    vector<Shape> computeShape() const override;

    int numInputs() const override { return 2; }
    int numOutputs() const override { return 1; }

    Tensor getBias() const { return inputs[2]; }
    void setAct(ActType act) { this->args.act = act; }
    ActType getAct() const { return args.act; }
    bool getTransA() const { return args.transA; }
    bool getTransB() const { return args.transB; }

    MatmulArgs getArgs() const { return args; }

  private:
    // Q: whether to check the output? Since we can build an Op first and then
    // assure output.
    // Fix 1: make shape inference a static method.
    bool checkValid(const TensorVec &inputs) const;
};

} // namespace it
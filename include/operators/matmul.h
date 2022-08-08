#pragma once
#include "core/operator.h"

namespace infini {

class MatmulNode : public OperatorNode {
  public:
    struct MatmulArgs : public OpAttrs {
        int b, m, n, k;
        // PET assume a row-major tensor layout. transA=false means default
        // dims, true means A should be transposed before matmul. This is in
        // oppsite to column-major BLAS.
        bool transA, transB;
        ActType act;

        MatmulArgs(int b, int m, int n, int k, bool transA, bool transB,
                   ActType act)
            : b(b), m(m), n(n), k(k), transA(transA), transB(transB), act(act) {
        }

        bool operator<(const OpAttrs &rhsGeneric) {
            auto rhs = dynamic_cast<const MatmulArgs &>(rhsGeneric);
            return std::tie(b, m, n, k, transA, transB, act) <
                   std::tie(rhs.b, rhs.m, rhs.n, rhs.k, rhs.transA, rhs.transB,
                            rhs.act);
        }
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
    OpAttrs getOpAttrs() const override { return args; }

  private:
    // Q: whether to check the output? Since we can build an Op first and then
    // assure output.
    // Fix 1: make shape inference a static method. But OpAttrs are required.
    bool checkValid(const TensorVec &inputs) const;
};

} // namespace infini
#pragma once
#include "core/operator.h"

namespace infini {

class MatmulNode : public OperatorNode {
  private:
    // InfiniTensor assume a row-major tensor layout. transA=false means default
    // dims, true means A should be transposed before matmul. This is in
    // oppsite to column-major BLAS.
    bool transA, transB;
    ActType act;

    // Auxiliary attributes
    int b, m, n, k;

  public:
    MatmulNode(Tensor A, Tensor B, Tensor C, bool transA = false,
               bool transB = false, Tensor bias = nullptr,
               ActType act = ActType::None);

    std::string toString() const override;
    vector<Shape> computeShape() const override;

    int numInputs() const override { return 2; }
    int numOutputs() const override { return 1; }

    Tensor getBias() const { return inputs[2]; }
    ActType getAct() const { return act; }
    bool getTransA() const { return transA; }
    bool getTransB() const { return transB; }
    int getB() const { return b; }
    int getM() const { return m; }
    int getN() const { return n; }
    int getK() const { return k; }

    HashType hashWithShape() const override;
    OpPerfKey getOpAttrs() const override;

  private:
    // Q: whether to check the output? Since we can build an Op first and then
    // assure output.
    // Fix 1: make shape inference a static method. But OpPerfKey are required.
    bool checkValid(const TensorVec &inputs) const;
};

} // namespace infini

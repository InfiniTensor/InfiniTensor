#pragma once
#include "core/operator.h"

namespace infini {
/**
 * @brief Matrix multiplication.
 *
 */
class MatmulObj : public OperatorObj {
  private:
    // InfiniTensor assumes a row-major tensor layout. `transA`=false means
    // default dims, true means A should be transposed before matmul. This is in
    // oppsite to the column-major BLAS.
    bool transA, transB;
    ActType act;

    // Auxiliary attributes which are not a part of operator attributes.
    int b, m, n, k;

  public:
    /**
     * @brief Matmul operator with batch broadcast and tensor transpose
     * supports. Only one tensor with singe batch can be broadcasted due to the
     * BLAS interface restriction. Tranpose indicates whether the last two
     * dimensions should be transposed before Matmul and does not affect other
     * leading dimensions.
     *
     * Matmul show how operators are defined in InfiniTensor. The constructor of
     * an operator can create output tensors for the operator or not, which
     * depends on `graph`.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param A The input tensor.
     * @param B The input tensor.
     * @param C C is the output of Matmul. If outputs are going to be created in
     * the constructor, C should be an empty Ref.
     * @param transA If matrix A should be transposed when computing.
     * @param transB If matrix B should be transposed when computing.
     * @param bias The bias tensor.
     * @param act The activation function.
     */
    MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C,
              bool transA = false, bool transB = false, Tensor bias = nullptr,
              ActType act = ActType::None);
    OP_CLONE(MatmulObj);

    std::string toString() const override;
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    int numInputs() const override { return inputs.size(); }
    int numOutputs() const override { return 1; }

    Tensor getBias() const { return inputs.size() > 2 ? inputs[2] : nullptr; }
    ActType getAct() const { return act; }
    auto getBMNKTransAB() const { return tuple(b, m, n, k, transA, transB); }
    bool getTransA() const { return transA; }
    bool getTransB() const { return transB; }
    int getB() const { return b; }
    int getM() const { return m; }
    int getN() const { return n; }
    int getK() const { return k; }
    auto getBMNK() const { return tuple{b, m, n, k}; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

} // namespace infini

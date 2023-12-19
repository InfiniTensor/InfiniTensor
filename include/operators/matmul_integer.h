#pragma once
#include "core/operator.h"

namespace infini {
/**
 * @brief Matrix multiplication.
 *
 */
class MatmulIntegerObj : public OperatorObj {
  private:
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
     * @param a_zero_point Zero point tensor for input 'A'.
     * @param b_zero_point Zero point tensor for input 'B'.
     */
    MatmulIntegerObj(GraphObj *graph, Tensor A, Tensor B, Tensor C,
                     Tensor a_zero_point = nullptr,
                     Tensor b_zero_point = nullptr);
    OP_CLONE(MatmulIntegerObj);

    std::string toString() const override;
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;

    int numInputs() const override { return inputs.size(); }
    int numOutputs() const override { return 1; }

    Tensor getZeroPointA() const {
        return inputs.size() > 2 ? inputs[2] : nullptr;
    }
    Tensor getZeroPointB() const {
        return inputs.size() > 3 ? inputs[3] : nullptr;
    }
    int getB() const { return b; }
    int getM() const { return m; }
    int getN() const { return n; }
    int getK() const { return k; }
    auto getBMNK() const { return tuple{b, m, n, k}; }
    vector<DataType> inferDataType(const TensorVec &inputs) const override;

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

} // namespace infini

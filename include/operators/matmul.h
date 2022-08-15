#pragma once
#include "core/operator.h"

namespace infini {

class MatmulNode : public OperatorNode {
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
     * @brief This comments show how operators is defined in InfiniTensor. The
     * constructor can create output tensors for the operator or not, which
     * depends on `graph`.
     *
     * @param graph If graph is not empty, create outputs in the constructor.
     * Otherwise, check the provided shape with the results of `inferShape` in
     * `checkValid`.
     * @param C C is the output of Matmul. If outputs are going to be created in
     * the constructor, C should be an empty Ref.
     */
    MatmulNode(GraphNode *graph, Tensor A, Tensor B, Tensor C,
               bool transA = false, bool transB = false, Tensor bias = nullptr,
               ActType act = ActType::None);

    std::string toString() const override;
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    int numInputs() const override { return 3; }
    int numOutputs() const override { return 1; }

    Tensor getBias() const { return inputs[2]; }
    ActType getAct() const { return act; }
    bool getTransA() const { return transA; }
    bool getTransB() const { return transB; }
    int getB() const { return b; }
    int getM() const { return m; }
    int getN() const { return n; }
    int getK() const { return k; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

} // namespace infini

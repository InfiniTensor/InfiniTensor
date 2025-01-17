#pragma once
#include "core/operator.h"

namespace infini {
/**
 * @brief General Matrix multiplication.
 *
 */
class GemmObj : public OperatorObj {
  private:
    // InfiniTensor assumes a row-major tensor layout. `transA`=false means
    // default dims, true means A should be transposed before matmul. This is in
    // oppsite to the column-major BLAS.
    float alpha, beta;
    bool transA, transB;

  public:
    /**
     * @brief Construct a new Gemm object.
     * @param graph The computation graph that this operator belongs to.
     * @param A The input tensor.
     * @param B The input tensor.
     * @param Y Y is the output of Matmul. If outputs are going to be created in
     * the constructor, Y should be an empty Ref.
     * @param C The bias tensor.
     * @param transA If matrix A should be transposed when computing.
     * @param transB If matrix B should be transposed when computing.
     */
    GemmObj(GraphObj *graph, Tensor A, Tensor B, Tensor Y, Tensor C,
            float alpha = 1.0f, float beta = 1.0f, bool transA = false,
            bool transB = false);
    OP_CLONE(GemmObj);

    std::string toString() const override;
    ~GemmObj() override {
        if (opDesc) {
            try {
                CHECK_ERROR(infiniopDestroyGEMMDescriptor(
                    (infiniopGEMMDescriptor_t)opDesc));
            } catch (const std::exception &e) {
                std::cerr << "Error in ~GemmObj: " << e.what() << std::endl;
            }
        }
    }

    void initInfiniOp(const Runtime context) override;
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;

    int numInputs() const override { return inputs.size(); }
    int numOutputs() const override { return 1; }

    bool getTransA() const { return transA; }
    bool getTransB() const { return transB; }
    float getAlpha() const { return alpha; }
    float getBeta() const { return beta; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

} // namespace infini

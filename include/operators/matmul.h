#pragma once
#include "core/operator.h"
#include "core/kernel.h"
#include "cuda/cuda_runtime.h"
#include <chrono>
#include <functional>

namespace infini {
struct MatmulCudnnPerfRecord : public PerfRecord {
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
    json to_json () override {
        return json {{"time", this->time}, {"algo", this->algo}};
    }
    void from_json (json j) override {
        j.at("time").get_to(this->time);
        j.at("algo").get_to(this->algo);
    }
};
constexpr int N_ALGO = 24;
constexpr cublasGemmAlgo_t ALGOS[N_ALGO] = {
    CUBLAS_GEMM_ALGO0,  CUBLAS_GEMM_ALGO1,  CUBLAS_GEMM_ALGO2,
    CUBLAS_GEMM_ALGO3,  CUBLAS_GEMM_ALGO4,  CUBLAS_GEMM_ALGO5,
    CUBLAS_GEMM_ALGO6,  CUBLAS_GEMM_ALGO7,  CUBLAS_GEMM_ALGO8,
    CUBLAS_GEMM_ALGO9,  CUBLAS_GEMM_ALGO10, CUBLAS_GEMM_ALGO11,
    CUBLAS_GEMM_ALGO12, CUBLAS_GEMM_ALGO13, CUBLAS_GEMM_ALGO14,
    CUBLAS_GEMM_ALGO15, CUBLAS_GEMM_ALGO16, CUBLAS_GEMM_ALGO17,
    CUBLAS_GEMM_ALGO18, CUBLAS_GEMM_ALGO19, CUBLAS_GEMM_ALGO20,
    CUBLAS_GEMM_ALGO21, CUBLAS_GEMM_ALGO22, CUBLAS_GEMM_ALGO23,
};
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
    MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C,
              bool transA = false, bool transB = false, Tensor bias = nullptr,
              ActType act = ActType::None);

    std::string toString() const override;
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    int numInputs() const override { return 2; }
    int numOutputs() const override { return 1; }

    Tensor getBias() const { return inputs[2]; }
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

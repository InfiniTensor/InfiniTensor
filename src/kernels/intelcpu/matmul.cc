#include "operators/matmul.h"
#include "core/kernel.h"
#include "intelcpu/mkl_runtime.h"

namespace infini {
template <typename T> class MklMatmul : public CpuKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<MatmulObj>(_op);
        IT_ASSERT(op->getInputs().size() == 2, "Bias is not supported yet.");
        const T *A = op->getInputs(0)->getRawDataPtr<T *>();
        const T *B = op->getInputs(1)->getRawDataPtr<T *>();
        T *C = op->getOutput()->getRawDataPtr<T *>();
        IT_ASSERT(op->getAct() == ActType::None);
        const int m = op->getM(), n = op->getN(), k = op->getK(),
                  b = op->getB();

        auto opA = op->getTransA() ? CblasTrans : CblasNoTrans;
        auto opB = op->getTransB() ? CblasTrans : CblasNoTrans;
        // lda is always a.col, and ldb is always b.col when row major
        const int lda = std::max((opA == CblasNoTrans) ? k : m, 1);
        const int ldb = std::max((opB == CblasNoTrans) ? n : k, 1);
        const int ldc = std::max(n, 1);

        const float alpha = 1.f, beta = 0.f;
        // TODO: Intel MKL ERROR will occur when using cblas_sgemm_batch
        for (int i = 0; i < b; ++i) {
            cblas_sgemm(CblasRowMajor, opA, opB, m, n, k, alpha, A + m * k * i,
                        lda, B + k * n * i, ldb, beta, C + m * n * i, ldc);
        }
    }
};

/*REGISTER_KERNEL(Device::INTELCPU, OpType::Matmul, DataType::Float32,
                MklMatmul<float>, "MklMatmul_CPU_float32");*/

} // namespace infini

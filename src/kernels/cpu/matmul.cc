#include "operators/matmul.h"
#include "core/kernel.h"

namespace infini {

template <typename T> class NaiveMatmul : public CpuKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<MatmulObj>(_op);
        IT_ASSERT(op->getInputs().size() == 2, "Bias is not supported yet.");
        T *A = op->getInputs(0)->getRawDataPtr<T *>();
        T *B = op->getInputs(1)->getRawDataPtr<T *>();
        T *C = op->getOutput()->getRawDataPtr<T *>();
        IT_ASSERT(op->getTransA() == false && op->getTransB() == false);
        IT_ASSERT(op->getAct() == ActType::None);
        IT_ASSERT(op->getB() == 1);
        const int M = op->getM(), N = op->getN(), K = op->getK();
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                C[i * N + j] = 0;
                for (int k = 0; k < K; k++) {
                    C[i * N + j] += A[i * K + k] * B[k * N + j];
                }
            }
        }
    }
};

REGISTER_KERNEL(Device::CPU, OpType::Matmul, DataType::UInt32,
                NaiveMatmul<uint32_t>, "MatmulNaive_CPU_uint32");
REGISTER_KERNEL(Device::CPU, OpType::Matmul, DataType::Float32,
                NaiveMatmul<float>, "MatmulNaive_CPU_float32");

} // namespace infini
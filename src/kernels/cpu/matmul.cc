#include "operators/matmul.h"
#include "core/kernel.h"

namespace infini {

class NaiveMatmul : public CpuKernelWithoutConfig {
    template <typename T>
    void doCompute(const Operator &_op, const RuntimeObj *context) const {
        auto op = as<MatmulObj>(_op);
        IT_ASSERT(op->getInputs().size() == 2, "Bias is not supported yet.");
        T *A = op->getInputs(0)->getRawDataPtr<T *>();
        T *B = op->getInputs(1)->getRawDataPtr<T *>();
        T *C = op->getOutput()->getRawDataPtr<T *>();
        IT_ASSERT(op->getTransA() == false && op->getTransB() == false);
        IT_ASSERT(op->getAct() == ActType::None);
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

    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
#define CASE(N)                                                                \
    case N:                                                                    \
        doCompute<DT<N>::t>(_op, context)

        int dataTypeIdx = _op->getDType().getIndex();
        switch (dataTypeIdx) {
            CASE(1); // DataType::Float32
            break;
            CASE(12); // DataType::UInt32
            break;
        default:
            IT_TODO_HALT();
        }
    }
};

REGISTER_KERNEL(Device::CPU, OpType::MatMul, NaiveMatmul, "MatmulNaive_CPU");

} // namespace infini

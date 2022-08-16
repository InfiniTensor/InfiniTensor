#include "operators/matmul.h"
#include "core/kernel.h"

namespace infini {

template <typename T> class NaiveMatmul : public Kernel {
    void compute(const Operator &_op, const PerfRecord &record) const override {
        auto op = as<MatmulObj>(_op);
        T *A = reinterpret_cast<T *>(op->getInputs(0)->getDataPtr().get());
        T *B = reinterpret_cast<T *>(op->getInputs(1)->getDataPtr().get());
        T *C = reinterpret_cast<T *>(op->getOutput()->getDataPtr().get());
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

    void compute(const Operator &op) const override { compute(op, {}); }

    PerfRecord tune(const Operator &op) const override {
        return PerfRecord{.time = timeit([this, &op]() { compute(op); })};
    }
};

REGISTER_KERNEL(Device::CPU, OpType::Matmul, DataType::UInt32,
                NaiveMatmul<uint32_t>, "MatmulNaive_CPU_uint32");
REGISTER_KERNEL(Device::CPU, OpType::Matmul, DataType::Float32,
                NaiveMatmul<float>, "MatmulNaive_CPU_float32");

} // namespace infini
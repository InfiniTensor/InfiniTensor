#include "operators/matmul.h"
#include "core/kernel.h"

namespace infini {

template <typename T> class NaiveMatmul : public Kernel {
    void compute(const Operator &_op) const override {
        auto op = as<MatmulNode>(_op);
        T *A = reinterpret_cast<T *>(op->getInputs(0)->getDataPtr().get());
        T *B = reinterpret_cast<T *>(op->getInputs(1)->getDataPtr().get());
        T *C = reinterpret_cast<T *>(op->getOutput()->getDataPtr().get());
        const auto args = op->getArgs();
        IT_ASSERT(args.transA == false && args.transB == false);
        IT_ASSERT(args.act == ActType::None);
        const int M = args.m, N = args.n, K = args.k;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                C[i * N + j] = 0;
                for (int k = 0; k < K; k++) {
                    C[i * N + j] += A[i * K + k] * B[k * N + j];
                }
            }
        }
    }

    void compute(const Operator &op, const PerfRecord &record) const override {
        compute(op);
    }

    PerfRecord tune(const Operator &op) const override {
        return PerfRecord{.time = timeit([this, &op]() { compute(op); })};
    }
};

REGISTER_KERNEL(Device::CPU, OpType::Matmul, DataType::Int32,
                NaiveMatmul<uint32_t>);
REGISTER_KERNEL(Device::CPU, OpType::Matmul, DataType::Float32,
                NaiveMatmul<float>);

} // namespace infini
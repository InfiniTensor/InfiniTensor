#include "operators/matmul.h"
#include "kunlun/kunlun_kernel_without_config.h"
#include "kunlun/kunlun_runtime.h"

namespace infini {
class MatmulXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<MatmulObj>(_op);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);
        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        bool transA = op->getTransA();
        bool transB = op->getTransB();

        auto b = op->getB();
        auto m = op->getM();
        auto n = op->getN();
        auto k = op->getK();

        auto ret = baidu::xpu::api::fc_batched<float, float, float, float>(
            context->KUNLUNHandle(), b, transA, transB, m, n, k, 1.0,
            (float *)aData, m * k, (float *)bData, n * k, 0.0, (float *)cData,
            m * n, nullptr, nullptr);
        assert(ret == 0);
        return;
    }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::MatMul, DataType::Float32, MatmulXdnn,
                "Matmul_xdnn_KUNLUN_Float32");
}; // namespace infini

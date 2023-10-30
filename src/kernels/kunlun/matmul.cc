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
        if (op->getInputs(0)->getDims().size() != 2 ||
            op->getInputs(1)->getDims().size() != 2) {
            IT_TODO_HALT();
        }

        auto m = transA ? op->getInputs(0)->getDims()[1]
                        : op->getInputs(0)->getDims()[0];
        auto n = transB ? op->getInputs(1)->getDims()[0]
                        : op->getInputs(1)->getDims()[1];
        auto k = transA ? op->getInputs(0)->getDims()[0]
                        : op->getInputs(0)->getDims()[1];

        auto ret = baidu::xpu::api::fc<float, float, float, int>(
            context->KUNLUNHandle(), (float *)aData, (float *)bData,
            (float *)cData, m, n, k, transA, transB, nullptr, nullptr, nullptr);
        assert(ret == 0);
        return;
    }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::MatMul, DataType::Float32, MatmulXdnn,
                "Matmul_xdnn_KUNLUN_Float32");
}; // namespace infini

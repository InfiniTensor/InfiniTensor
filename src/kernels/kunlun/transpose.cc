#include "operators/transpose.h"
#include "kunlun/kunlun_kernel_without_config.h"
#include "kunlun/kunlun_runtime.h"

namespace infini {
class TransposeXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<TransposeObj>(_op);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto dimin = op->getInputs(0)->getDims();
        auto permute = op->getPermute();

        if (dimin.size() != 4) {
            IT_TODO_HALT();
        }

        auto ret = baidu::xpu::api::transpose<float>(
            context->KUNLUNHandle(), (float *)aData, (float *)cData, dimin,
            permute);
        assert(ret == 0);
        return;
    }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::Transpose, DataType::Float32,
                TransposeXdnn, "Transpose_xdnn_KUNLUN_Float32");
}; // namespace infini

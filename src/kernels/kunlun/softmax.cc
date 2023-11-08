#include "operators/softmax.h"
#include "kunlun/kunlun_kernel_without_config.h"
#include "kunlun/kunlun_runtime.h"

namespace infini {
class SoftmaxXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<SoftmaxObj>(_op);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);
        auto dim = op->getInputs(0)->getDims();
        auto axis = op->getAxis();

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto ret = baidu::xpu::api::softmax<float>(
            context->KUNLUNHandle(), (float *)aData, (float *)cData, dim, axis);
        // auto ret = baidu::xpu::api::relu<float>(
        //     context->KUNLUNHandle(), (float *)aData, (float *)cData, op->getInputs(0)->size());
        assert(ret == 0);
        return;
    }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::Softmax, DataType::Float32, SoftmaxXdnn,
                "Softmax_xdnn_KUNLUN_Float32");
}; // namespace infini

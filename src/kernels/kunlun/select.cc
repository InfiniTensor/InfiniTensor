#include "kunlun/kunlun_kernel_without_config.h"
#include "kunlun/kunlun_runtime.h"
#include "operators/where.h"

namespace infini {
class WhereXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<WhereObj>(_op);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getInputs(2)->getRawDataPtr<void *>());
        void *const dData = (op->getOutput()->getRawDataPtr<void *>());

        auto aDim = op->getInputs(0)->getDims();
        auto bDim = op->getInputs(1)->getDims();
        auto cDim = op->getInputs(2)->getDims();
        auto dDim = op->getOutput()->getDims();

        auto ret = baidu::xpu::api::select<float>(
            context->KUNLUNHandle(), (bool *)cData, (float *)aData,
            (float *)bData, (float *)dData, cDim, aDim);
        assert(ret == 0);
        return;
    }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::Where, DataType::Float32, WhereXdnn,
                "Where_xdnn_KUNLUN_Float32");
}; // namespace infini

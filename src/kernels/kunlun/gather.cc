#include "operators/gather.h"
#include "core/common.h"
#include "kunlun/kunlun_kernel_without_config.h"
#include "kunlun/kunlun_runtime.h"

namespace infini {
class GatherXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<GatherObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>()); // data
        void *const bData =
            (op->getInputs(1)->getRawDataPtr<void *>()); // indice
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        Shape aShape = op->getInputs(0)->getDims();
        Tensor bTensor = op->getInputs(1);
        int axis = op->getAxis();
        checkKUNLUNError((baidu::xpu::api::gather<float, int>(
            context->KUNLUNHandle(), (float *)aData, (int *)bData,
            (float *)cData, aShape, bTensor->size(), axis)));

        return;
    }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::Gather, GatherXdnn,
                "Gather_xdnn_KUNLUN");
}; // namespace infini

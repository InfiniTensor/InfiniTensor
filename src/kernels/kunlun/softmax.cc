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

        if (op->getDType() == DataType::Float32) {
            checkKUNLUNError(xdnn::softmax<float>(context->KUNLUNHandle(),
                                                  (float *)aData,
                                                  (float *)cData, dim, axis));
        } else if (op->getDType() == DataType::Float16) {
            checkKUNLUNError(xdnn::softmax<float16>(
                context->KUNLUNHandle(), (float16 *)aData, (float16 *)cData,
                dim, axis));
        } else {
            IT_ASSERT(false, "Unsupported data type");
        }

        return;
    }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::Softmax, SoftmaxXdnn,
                "Softmax_xdnn_KUNLUN");
}; // namespace infini

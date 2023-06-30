#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"
#include "operators/unary.h"
#include <cmath>

namespace infini {
class ClipCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ClipObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        float min = op->getMin();
        float max = op->getMax();

        cnnlTensorDescriptor_t aDesc;
        auto aDim = op->getInputs(0)->getDims();

        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, aDim.size(),
                                               aDim.data()));
        cnnlStatus_t stat =
            cnnlClip(context->cnnlHandle(), aDesc, aData,
                     std::isfinite(min) ? &min : nullptr,
                     std::isfinite(max) ? &max : nullptr, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::Clip, DataType::Float32, ClipCnnl,
                "Clip_cnnl_BANG_Float32");

}; // namespace infini

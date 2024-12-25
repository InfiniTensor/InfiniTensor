#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"
#include "operators/unary.h"

namespace infini {
class ClipCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ClipObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        void *const min = op->numInputs() > 1
                              ? (op->getInputs(1)->getRawDataPtr<void *>())
                              : nullptr;
        void *const max = op->numInputs() > 2
                              ? (op->getInputs(2)->getRawDataPtr<void *>())
                              : nullptr;

        cnnlTensorDescriptor_t aDesc;
        auto aDim = op->getInputs(0)->getDims();

        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            aDesc, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(op->getDType()),
            aDim.size(), aDim.data()));

        cnnlTensorDescriptor_t cDesc;
        auto cDim = op->getOutput()->getDims();

        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            cDesc, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(op->getDType()),
            cDim.size(), cDim.data()));

        cnnlStatus_t stat =
            cnnlClip_v2(context->cnnlHandle(), CNNL_POINTER_MODE_DEVICE, aDesc,
                        aData, min, max, cDesc, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::Clip, ClipCnnl, "Clip_cnnl_BANG");

}; // namespace infini

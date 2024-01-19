#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"
#include "operators/unary.h"

namespace infini {
class FillCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<FillObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        float value = op->getValue();

        cnnlTensorDescriptor_t cDesc;
        auto cDim = op->getOutput()->getDims();

        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            cDesc, CNNL_LAYOUT_NCHW, cnnlDataTypeConvert(op->getDType()),
            cDim.size(), cDim.data()));

        cnnlStatus_t stat =
            cnnlFill(context->cnnlHandle(), value, cDesc, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::Fill, FillCnnl, "Fill_cnnl_BANG");

}; // namespace infini

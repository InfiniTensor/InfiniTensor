#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"
#include "operators/unary.h"

namespace infini {
class HardtanhCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<HardtanhObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        float min = op->getMin();
        float max = op->getMax();

        cnnlTensorDescriptor_t aDesc;
        auto dim = op->getInputs(0)->getDims();

        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, dim.size(), dim.data()));

        cnnlStatus_t stat = cnnlHardtanh(context->cnnlHandle(), aDesc, aData,
                                         max, min, aDesc, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::Hardtanh, DataType::Float32, HardtanhCnnl,
                "Hardtanh_cnnl_BANG_Float32");

}; // namespace infini

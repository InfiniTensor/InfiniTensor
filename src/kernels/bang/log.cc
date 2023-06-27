#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"
#include "operators/unary.h"

namespace infini {
class LogCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<LogObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto type = op->getType();
        cnnlLogBase_t base;
        switch (type) {
        case LogObj::Log2:
            base = CNNL_LOG_2;
            break;
        case LogObj::LogE:
            base = CNNL_LOG_E;
            break;
        case LogObj::Log10:
            base = CNNL_LOG_10;
            break;
        default:
            IT_TODO_HALT();
        }

        cnnlTensorDescriptor_t aDesc, cDesc;
        auto aDim = op->getInputs(0)->getDims();
        auto cDim = op->getOutput()->getDims();

        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, aDim.size(),
                                               aDim.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, cDim.size(),
                                               cDim.data()));

        cnnlStatus_t stat =
            cnnlLog_v2(context->cnnlHandle(), CNNL_COMPUTATION_HIGH_PRECISION,
                       base, aDesc, aData, cDesc, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::Log, DataType::Float32, LogCnnl,
                "Log_cnnl_BANG_Float32");

}; // namespace infini

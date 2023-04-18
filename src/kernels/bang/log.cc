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
        auto dim = op->getInputs(0)->getDims();
        if (dim.size() != 4)
            IT_TODO_HALT();

        int dim_array[4] = {dim[0], dim[1], dim[2], dim[3]};
        // get inputs
        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));

        // get outputs
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));

        cnnlStatus_t stat =
            cnnlLog_v2(context->cnnlHandle(), CNNL_COMPUTATION_HIGH_PRECISION,
                       base, aDesc, aData, cDesc, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        // Destories in BANG does not require sync. But cnnl does not state
        // whether sync is required before destories.
        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::Log, DataType::Float32, LogCnnl,
                "Log_cnnl_BANG_Float32");

}; // namespace infini

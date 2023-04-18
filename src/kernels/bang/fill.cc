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
        auto dim = op->getOutput()->getDims();
        if (dim.size() != 4)
            IT_TODO_HALT();

        int dim_array[4] = {dim[0], dim[1], dim[2], dim[3]};
        // get outputs
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));

        cnnlStatus_t stat =
            cnnlFill(context->cnnlHandle(), value, cDesc, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        // Destories in BANG does not require sync. But cnnl does not state
        // whether sync is required before destories.
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::Fill, DataType::Float32, FillCnnl,
                "Fill_cnnl_BANG_Float32");

}; // namespace infini

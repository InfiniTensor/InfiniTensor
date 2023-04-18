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
        float min = op->getMin().value();
        float max = op->getMax().value();

        cnnlTensorDescriptor_t aDesc;
        auto dim = op->getInputs(0)->getDims();
        if (dim.size() != 4)
            IT_TODO_HALT();

        int dim_array[4] = {dim[0], dim[1], dim[2], dim[3]};
        // get inputs
        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));

        cnnlStatus_t stat =
            cnnlClip(context->cnnlHandle(), aDesc, aData, &min, &max, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        // Destories in BANG does not require sync. But cnnl does not state
        // whether sync is required before destories.
        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::Clip, DataType::Float32, ClipCnnl,
                "Clip_cnnl_BANG_Float32");

}; // namespace infini

#include "operators/reshape.h"
#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"

namespace infini {
class CopyBang : public BangKernelWithoutConfig {
    void compute(const Operator &op,
                 const RuntimeObj *_context) const override {
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);
        auto inData = op->getInputs(0)->getRawDataPtr<void *>();
        auto outData = op->getOutputs()[0]->getRawDataPtr<void *>();
        cnnlTensorDescriptor_t aDesc;
        auto dim = op->getInputs(0)->getDims();

        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_ARRAY,
                                               CNNL_DTYPE_FLOAT, dim.size(),
                                               dim.data()));
        cnnlStatus_t stat =
            cnnlCopy(context->cnnlHandle(), aDesc, inData, aDesc, outData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
    }
};
// reshape/flatten/identity all act as copying from input to output.
REGISTER_KERNEL(Device::BANG, OpType::Reshape, CopyBang, "Reshape_BANG");
REGISTER_KERNEL(Device::BANG, OpType::Flatten, CopyBang, "Flatten_BANG");
REGISTER_KERNEL(Device::BANG, OpType::Identity, CopyBang, "Identity_BANG");

} // namespace infini

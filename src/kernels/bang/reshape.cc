#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"
#include "operators/unary.h"

namespace infini {
class CopyBang : public BangKernelWithoutConfig {
    void compute(const Operator &op,
                 const RuntimeObj *_context) const override {
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);
        auto inData = op->getInputs(0)->getRawDataPtr<void *>();
        auto outData = op->getOutputs()[0]->getRawDataPtr<void *>();
        auto handle = context->cnnlHandle();
        cnrtQueue_t queue;
        cnnlGetQueue(handle, &queue);
        cnrtMemcpyAsync(outData, inData, op->getInputs(0)->getBytes(), queue,
                        cnrtMemcpyDevToDev);
    }
};
// reshape/flatten/identity all act as copying from input to output.
REGISTER_KERNEL(Device::BANG, OpType::Reshape, DataType::Float32, CopyBang,
                "Reshape_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::Flatten, DataType::Float32, CopyBang,
                "Flatten_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::Identity, DataType::Float32, CopyBang,
                "Identity_BANG_Float32");

} // namespace infini

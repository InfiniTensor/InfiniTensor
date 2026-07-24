#include "core/kernel.h"
#include "utils/infiniop_utils.h"

namespace infini {

class CopyOp : public Kernel {
    void compute(const Operator &op, const RuntimeObj *context) const override {
        auto inData = (op->getInputs(0)->getRawDataPtr<void *>());
        auto outData = (op->getOutput()->getRawDataPtr<void *>());
        // 此处应使用 async 拷贝
        context->copyBlobInsideRuntime(outData, inData,
                                       op->getInputs(0)->getBytes());
    }

    PerfRecord tune(const Operator &op,
                    const RuntimeObj *context) const override {
        // TODO: tune should be in infiniop
        return PerfRecord();
    }

    void compute(const Operator &op, const PerfRecord &record,
                 const RuntimeObj *context) const override {
        compute(op, context);
    }
};

// reshape/flatten/identity/squeeze/unsqueeze all act as copying from input to
// output.

REGISTER_KERNEL(Device::CPU, OpType::Reshape, CopyOp, "Reshape_infiniop_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Flatten, CopyOp, "Flatten_infiniop_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Identity, CopyOp, "Identity_infiniop_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Squeeze, CopyOp, "Squeeze_infiniop_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Unsqueeze, CopyOp,
                "Unsqueeze_infiniop_CPU");

REGISTER_KERNEL(Device::CUDA, OpType::Reshape, CopyOp, "Reshape_infiniop_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Flatten, CopyOp, "Flatten_infiniop_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Identity, CopyOp,
                "Identity_infiniop_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Squeeze, CopyOp, "Squeeze_infiniop_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Unsqueeze, CopyOp,
                "Unsqueeze_infiniop_CUDA");
}; // namespace infini

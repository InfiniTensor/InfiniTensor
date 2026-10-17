#include "operators/reshape.h"
#include "infiniops_common.h"
#include "operators/squeeze.h"
#include "operators/unsqueeze.h"

namespace infini {
namespace {

class CopyInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto runtime = dynamic_cast<const InfiniRuntimeObj *>(context);
        IT_ASSERT(runtime != nullptr);
        // A synchronous/default-queue copy need not wait for ATen work on
        // this runtime's queue (notably on MLU). Keep data movement ordered
        // with its producer and consumer, and usable inside graph capture.
        auto error = ::infini::rt::runtime::MemcpyAsync(
            _op->getOutput()->getRawDataPtr<void *>(),
            _op->getInputs(0)->getRawDataPtr<void *>(),
            _op->getOutput()->getBytes(),
            ::infini::rt::runtime::kMemcpyDeviceToDevice,
            reinterpret_cast<::infini::rt::runtime::Stream>(runtime->getStream()));
        IT_ASSERT(error == ::infini::rt::runtime::kSuccess,
                  "InfiniRT reshape stream copy failed");
    }
};

} // namespace

REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Reshape, CopyInfiniOps,
                "Reshape_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Flatten, CopyInfiniOps,
                "Flatten_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Identity, CopyInfiniOps,
                "Identity_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Squeeze, CopyInfiniOps,
                "Squeeze_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Unsqueeze, CopyInfiniOps,
                "Unsqueeze_InfiniOps");

} // namespace infini

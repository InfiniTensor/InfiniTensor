#include "operators/reshape.h"
#include "infiniops_common.h"
#include "operators/squeeze.h"
#include "operators/unsqueeze.h"

namespace infini {
namespace {

class CopyInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        context->copyBlob(_op->getOutput().get(), _op->getInputs(0).get());
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

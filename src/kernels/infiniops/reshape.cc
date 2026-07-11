#include "infiniops_common.h"
#include "operators/reshape.h"
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

#ifdef USE_CUDA
REGISTER_KERNEL(Device::CUDA, OpType::Reshape, CopyInfiniOps,
                "Reshape_InfiniOps_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Flatten, CopyInfiniOps,
                "Flatten_InfiniOps_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Identity, CopyInfiniOps,
                "Identity_InfiniOps_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Squeeze, CopyInfiniOps,
                "Squeeze_InfiniOps_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Unsqueeze, CopyInfiniOps,
                "Unsqueeze_InfiniOps_CUDA");
#endif

} // namespace infini

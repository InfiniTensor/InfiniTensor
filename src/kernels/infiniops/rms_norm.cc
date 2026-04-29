#include "core/kernel.h"
#include "core/infiniops_bridge/tensor_convert.h"
#include "cpu/rms_norm/rms_norm.h"
#include "operators/rms_norm.h"

namespace infini {

class RMSNormInfiniOpsKernel : public KernelWithoutConfig {
    void compute(const Operator &op, const RuntimeObj *context) const override {
        auto normOp = as<RMSNormObj>(op);

        auto input = toInfiniOpsTensor(normOp->getInputs(0).get());
        auto weight = toInfiniOpsTensor(normOp->getInputs(1).get());
        auto output = toInfiniOpsTensor(normOp->getOutput().get());

        infini::ops::Handle handle;
        infini::ops::Config config;

        // Use default eps (1e-6) — InfiniTensor's RMSNormObj doesn't expose eps
        infini::ops::RmsNorm::Call(handle, config, input, weight, output);
    }
};

REGISTER_KERNEL(Device(Device::Type::kCpu), OpType::RMSNorm,
                RMSNormInfiniOpsKernel, "RMSNorm_InfiniOps_CPU");

} // namespace infini

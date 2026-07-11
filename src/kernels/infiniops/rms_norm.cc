#include "infiniops_common.h"
#include "operators/rms_norm.h"

#include <base/rms_norm.h>

namespace infini {
namespace {

class RMSNormInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<RMSNormObj>(_op);

        auto input = op->getInputs(0);
        auto weight = op->getInputs(1);
        auto output = op->getOutput();
        const auto &inputShape = input->getDims();
        IT_ASSERT(!inputShape.empty());
        const auto hiddenSize = inputShape.back();
        IT_ASSERT(hiddenSize == static_cast<int>(weight->size()));

        auto handle = infiniops::makeInfiniOpsHandle(context);
        auto inputView = infiniops::toInfiniOpsTensor(input, context);
        auto weightView = infiniops::toInfiniOpsTensor(weight, context);
        auto outputView = infiniops::toInfiniOpsTensor(output, context);

        ::infini::ops::RmsNorm::Call(handle, {}, inputView, weightView, 1e-6f,
                                     outputView);
    }
};

} // namespace

REGISTER_KERNEL(Device::CPU, OpType::RMSNorm, RMSNormInfiniOps,
                "RMSNorm_InfiniOps_CPU");

#ifdef USE_CUDA
REGISTER_KERNEL(Device::CUDA, OpType::RMSNorm, RMSNormInfiniOps,
                "RMSNorm_InfiniOps_CUDA");
#endif

#ifdef USE_BANG
REGISTER_KERNEL(Device::BANG, OpType::RMSNorm, RMSNormInfiniOps,
                "RMSNorm_InfiniOps_BANG");
#endif

#ifdef USE_INTELCPU
REGISTER_KERNEL(Device::INTELCPU, OpType::RMSNorm, RMSNormInfiniOps,
                "RMSNorm_InfiniOps_INTELCPU");
#endif

} // namespace infini

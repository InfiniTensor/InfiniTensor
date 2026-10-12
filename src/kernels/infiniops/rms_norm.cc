#include "operators/rms_norm.h"
#include "infiniops_common.h"
#include "infiniops_dispatch.h"

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

        auto config =
            infiniops::makeInfiniOpsConfig<::infini::ops::RmsNorm>(context);
        infiniops::dispatch::callRmsNorm(handle, config, inputView, weightView,
                                         1e-6f, outputView);
    }
};

} // namespace

REGISTER_KERNEL(ExecutionProvider::NativeCpu, OpType::RMSNorm, RMSNormInfiniOps,
                "RMSNorm_InfiniOps_CPU");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::RMSNorm, RMSNormInfiniOps,
                "RMSNorm_InfiniOps");

} // namespace infini

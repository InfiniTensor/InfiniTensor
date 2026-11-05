#include "operators/instance_norm.h"
#include "infiniops_common.h"
#include "infiniops_dispatch.h"

#include <base/instance_norm.h>

namespace infini {
namespace {

class InstanceNormInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op, const RuntimeObj *context) const override {
        auto op = as<InstanceNormObj>(_op);
        auto input = infiniops::toInfiniOpsTensor(op->getInputs(0), context);
        auto scale = infiniops::toInfiniOpsTensor(op->getInputs(1), context);
        auto bias = infiniops::toInfiniOpsTensor(op->getInputs(2), context);
        auto out = infiniops::toInfiniOpsTensor(op->getOutput(), context);
        auto handle = infiniops::makeInfiniOpsHandle(context);
        auto config = infiniops::makeInfiniOpsConfig<::infini::ops::InstanceNorm>(context);
        infiniops::dispatch::callInstanceNorm(handle, config, input, scale, bias,
                                             op->getEps(), out);
    }
};

} // namespace

REGISTER_KERNEL(ExecutionProvider::Infini, OpType::InstanceNormalization,
                InstanceNormInfiniOps, "InstanceNorm_InfiniOps");

} // namespace infini

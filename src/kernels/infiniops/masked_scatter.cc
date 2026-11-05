#include "operators/masked_scatter.h"
#include "infiniops_common.h"
#include "infiniops_dispatch.h"

#include <base/masked_scatter.h>

namespace infini {
namespace {

class MaskedScatterInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &op, const RuntimeObj *context) const override {
        auto input = infiniops::toInfiniOpsTensor(op->getInputs(0), context);
        auto mask = infiniops::toInfiniOpsTensor(op->getInputs(1), context);
        auto source = infiniops::toInfiniOpsTensor(op->getInputs(2), context);
        auto out = infiniops::toInfiniOpsTensor(op->getOutput(), context);
        auto handle = infiniops::makeInfiniOpsHandle(context);
        auto config = infiniops::makeInfiniOpsConfig<::infini::ops::MaskedScatter>(context);
        infiniops::dispatch::callMaskedScatter(handle, config, input, mask, source, out);
    }
};

} // namespace

REGISTER_KERNEL(ExecutionProvider::Infini, OpType::MaskedScatter,
                MaskedScatterInfiniOps, "MaskedScatter_InfiniOps");

} // namespace infini

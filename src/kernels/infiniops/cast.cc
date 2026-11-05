#include "infiniops_common.h"
#include "infiniops_dispatch.h"
#include "operators/unary.h"

#include <base/cast.h>
#include <base/copy.h>

namespace infini {
namespace {

class CastInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op, const RuntimeObj *context) const override {
        auto input = infiniops::toInfiniOpsTensor(_op->getInputs(0), context);
        auto output = infiniops::toInfiniOpsTensor(_op->getOutput(), context);
        auto handle = infiniops::makeInfiniOpsHandle(context);
        if (input.dtype() == output.dtype()) {
            auto config = infiniops::makeInfiniOpsConfig<::infini::ops::Copy>(context);
            infiniops::dispatch::callCopy(handle, config, input, true, output);
        } else {
            auto config = infiniops::makeInfiniOpsConfig<::infini::ops::Cast>(context);
            infiniops::dispatch::callCast(handle, config, input, output);
        }
    }
};

} // namespace

REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Cast, CastInfiniOps,
                "Cast_InfiniOps");
} // namespace infini

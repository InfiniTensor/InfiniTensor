#include "infiniops_common.h"
#include "infiniops_dispatch.h"
#include "operators/reduce.h"

#include <base/argmax.h>

namespace infini {
namespace {

class ArgMaxInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op, const RuntimeObj *context) const override {
        auto op = as<ArgMaxObj>(_op);
        auto input = infiniops::toInfiniOpsTensor(op->getInputs(0), context);
        auto output = infiniops::toInfiniOpsTensor(op->getOutput(), context);
        auto handle = infiniops::makeInfiniOpsHandle(context);
        auto config = infiniops::makeInfiniOpsConfig<::infini::ops::Argmax>(context);
        infiniops::dispatch::callArgmax(handle, config, input,
                                       op->getAxis(), op->getKeepDims(), output);
    }
};

} // namespace

REGISTER_KERNEL(ExecutionProvider::Infini, OpType::ArgMax, ArgMaxInfiniOps,
                "ArgMax_InfiniOps");
} // namespace infini

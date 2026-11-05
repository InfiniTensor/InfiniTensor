#include "operators/scatter_elements.h"
#include "infiniops_common.h"
#include "infiniops_dispatch.h"

#include <base/scatter.h>

namespace infini {
namespace {

class ScatterElementsInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op, const RuntimeObj *context) const override {
        auto op = as<ScatterElementsObj>(_op);
        auto input = infiniops::toInfiniOpsTensor(op->getInputs(0), context);
        auto index = infiniops::toInfiniOpsTensor(op->getInputs(1), context);
        auto updates = infiniops::toInfiniOpsTensor(op->getInputs(2), context);
        auto output = infiniops::toInfiniOpsTensor(op->getOutput(), context);
        auto handle = infiniops::makeInfiniOpsHandle(context);
        auto config = infiniops::makeInfiniOpsConfig<::infini::ops::Scatter>(context);
        infiniops::dispatch::callScatter(handle, config, input, index, updates,
                                         op->getAxis(), op->getReduction(), output);
    }
};

} // namespace

REGISTER_KERNEL(ExecutionProvider::Infini, OpType::ScatterElements,
                ScatterElementsInfiniOps, "ScatterElements_InfiniOps");

} // namespace infini

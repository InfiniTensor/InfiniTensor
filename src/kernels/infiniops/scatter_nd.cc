#include "operators/scatter_nd.h"
#include "infiniops_common.h"
#include "infiniops_dispatch.h"

#include <base/scatter_nd.h>

namespace infini {
namespace {

class ScatterNdInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op, const RuntimeObj *context) const override {
        auto op = as<ScatterNdObj>(_op);
        auto input = infiniops::toInfiniOpsTensor(op->getInputs(0), context);
        auto indices = infiniops::toInfiniOpsTensor(op->getInputs(1), context);
        auto updates = infiniops::toInfiniOpsTensor(op->getInputs(2), context);
        auto output = infiniops::toInfiniOpsTensor(op->getOutput(), context);
        auto handle = infiniops::makeInfiniOpsHandle(context);
        auto config = infiniops::makeInfiniOpsConfig<::infini::ops::ScatterNd>(context);
        infiniops::dispatch::callScatterNd(handle, config, input, indices,
                                          updates, op->isAdd(), output);
    }
};

} // namespace

REGISTER_KERNEL(ExecutionProvider::Infini, OpType::ScatterND,
                ScatterNdInfiniOps, "ScatterNd_InfiniOps");

} // namespace infini

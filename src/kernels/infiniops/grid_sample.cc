#include "operators/grid_sample.h"
#include "infiniops_common.h"
#include "infiniops_dispatch.h"

#include <base/grid_sample.h>

namespace infini {
namespace {

class GridSampleInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op, const RuntimeObj *context) const override {
        const auto op = as<GridSampleObj>(_op);
        const auto [nearest, reflection, alignCorners] = op->getAttributes();
        auto input = infiniops::toInfiniOpsTensor(op->getInputs(0), context);
        auto grid = infiniops::toInfiniOpsTensor(op->getInputs(1), context);
        auto out = infiniops::toInfiniOpsTensor(op->getOutput(), context);
        auto handle = infiniops::makeInfiniOpsHandle(context);
        auto config = infiniops::makeInfiniOpsConfig<::infini::ops::GridSample>(context);
        infiniops::dispatch::callGridSample(handle, config, input, grid, nearest,
                                          reflection, alignCorners, out);
    }
};

} // namespace

REGISTER_KERNEL(ExecutionProvider::Infini, OpType::GridSample,
                GridSampleInfiniOps, "GridSample_InfiniOps");

} // namespace infini

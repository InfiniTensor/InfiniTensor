#include "operators/roi_align.h"
#include "infiniops_common.h"
#include "infiniops_dispatch.h"

#include <base/roi_align.h>

namespace infini {
namespace {

class RoiAlignInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op, const RuntimeObj *context) const override {
        const auto op = as<RoiAlignObj>(_op);
        const auto [samplingRatio, spatialScale, aligned] = op->getAttributes();
        auto input = infiniops::toInfiniOpsTensor(op->getInputs(0), context);
        auto boxes = infiniops::toInfiniOpsTensor(op->getInputs(1), context);
        auto out = infiniops::toInfiniOpsTensor(op->getOutput(), context);
        auto handle = infiniops::makeInfiniOpsHandle(context);
        auto config = infiniops::makeInfiniOpsConfig<::infini::ops::RoiAlign>(context);
        infiniops::dispatch::callRoiAlign(handle, config, input, boxes,
                                         samplingRatio, spatialScale, aligned, out);
    }
};

} // namespace

REGISTER_KERNEL(ExecutionProvider::Infini, OpType::RoiAlign, RoiAlignInfiniOps,
                "RoiAlign_InfiniOps");

} // namespace infini

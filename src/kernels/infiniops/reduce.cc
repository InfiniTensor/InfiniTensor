#include "operators/reduce.h"
#include "infiniops_common.h"
#include "infiniops_dispatch.h"

#ifdef USE_INFINIOPS_ATEN_KERNELS
#include <base/mean.h>
#endif
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace infini {
namespace {

#ifdef USE_INFINIOPS_ATEN_KERNELS
class ReduceMeanAtenInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<ReduceMeanObj>(_op);
        std::vector<int64_t> axes;
        axes.reserve(op->getAxes().size());
        for (const auto axis : op->getAxes()) {
            axes.push_back(static_cast<int64_t>(axis));
        }

        auto input = infiniops::toInfiniOpsTensor(op->getInputs(0), context);
        auto output = infiniops::toInfiniOpsTensor(op->getOutput(), context);
        auto handle = infiniops::makeInfiniOpsHandle(context);
        auto config =
            infiniops::makeInfiniOpsAtenConfig<::infini::ops::Mean>(context);
        infiniops::dispatch::callMean(
            handle, config, input,
            std::optional<std::vector<int64_t>>{std::move(axes)},
            op->getKeepDims(), std::optional<::infini::rt::DataType>{}, output);
    }
};
#endif

} // namespace

#ifdef USE_INFINIOPS_ATEN_KERNELS
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::ReduceMean,
                ReduceMeanAtenInfiniOps, "ReduceMean_InfiniOps");
#endif

} // namespace infini

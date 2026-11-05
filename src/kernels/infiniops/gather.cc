#include "infiniops_common.h"
#include "infiniops_dispatch.h"
#include "operators/gather.h"

#include <base/index_select.h>
#include <base/gather_axis.h>

namespace infini {
namespace {

class GatherInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op, const RuntimeObj *context) const override {
        auto op = as<GatherObj>(_op);
        const auto data = op->getInputs(0);
        const auto index = op->getInputs(1);
        const auto result = op->getOutput();
        if (result->size() == 0)
            return;
        const auto axis = op->getAxis();
        const auto dims = data->getDims();
        IT_ASSERT(axis >= 0 && static_cast<size_t>(axis) < dims.size());
        // Gather inserts the entire index shape; IndexSelect writes the same
        // contiguous elements with those dimensions flattened into one axis.
        ::infini::rt::TensorView::Shape flatOutput(dims.size());
        for (size_t i = 0; i < dims.size(); ++i)
            flatOutput[i] = dims[i];
        flatOutput[axis] = index->size();
        auto input = infiniops::toInfiniOpsTensor(data, context);
        auto indices = infiniops::makeInfiniOpsTensor(
            index->getRawDataPtr<void *>(), {index->size()},
            infiniops::toInfiniOpsDataType(index->getDType()), context);
        auto output = infiniops::makeInfiniOpsTensor(
            result->getRawDataPtr<void *>(), flatOutput,
            infiniops::toInfiniOpsDataType(result->getDType()), context);
        auto handle = infiniops::makeInfiniOpsHandle(context);
        if (input.device().type() == ::infini::ops::Device::Type::kCambricon) {
            auto config =
                infiniops::makeInfiniOpsConfig<::infini::ops::GatherAxis>(context);
            infiniops::dispatch::callGatherAxis(handle, config, input, indices,
                                               axis, output);
        } else {
            auto config =
                infiniops::makeInfiniOpsConfig<::infini::ops::IndexSelect>(context);
            infiniops::dispatch::callIndexSelect(handle, config, input, indices,
                                               axis, output);
        }
    }
};

} // namespace

REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Gather, GatherInfiniOps,
                "Gather_InfiniOps_IndexSelect");
} // namespace infini

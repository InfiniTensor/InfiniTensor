#include "operators/reduce.h"
#include "infiniops_common.h"
#include "infiniops_dispatch.h"
#include <base/norm.h>

#include <base/mean.h>
#include <base/sum.h>
#include <base/amax.h>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace infini {
namespace {

class ReduceL2InfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op, const RuntimeObj *context) const override {
        auto op = as<ReduceL2Obj>(_op);
        auto input = infiniops::toInfiniOpsTensor(op->getInputs(0), context);
        auto output = infiniops::toInfiniOpsTensor(op->getOutput(), context);
        auto handle = infiniops::makeInfiniOpsHandle(context);
        auto config = infiniops::makeInfiniOpsConfig<::infini::ops::Norm>(context);
        auto &axes = op->getAxes();
        infiniops::dispatch::callNorm(handle, config, input, 2.0,
                                     {axes.begin(), axes.end()},
                                     op->getKeepDims(), output);
    }
};

class ReduceMeanInfiniOps : public infiniops::KernelWithoutConfig {
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
#ifdef USE_INFINIOPS_ATEN_KERNELS
        auto config = infiniops::makeInfiniOpsAtenConfig<::infini::ops::Mean>(context);
#else
        auto config = infiniops::makeInfiniOpsConfig<::infini::ops::Mean>(context);
#endif
        infiniops::dispatch::callMean(
            handle, config, input,
            std::optional<std::vector<int64_t>>{std::move(axes)},
            op->getKeepDims(), std::optional<::infini::rt::DataType>{}, output);
    }
};

class ReduceSumInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op, const RuntimeObj *context) const override {
        auto op = as<ReduceSumObj>(_op);
        auto input = infiniops::toInfiniOpsTensor(op->getInputs(0), context);
        auto output = infiniops::toInfiniOpsTensor(op->getOutput(), context);
        auto handle = infiniops::makeInfiniOpsHandle(context);
        auto config = infiniops::makeInfiniOpsConfig<::infini::ops::Sum>(context);
        const auto &axes = op->getAxes();
        infiniops::dispatch::callSum(handle, config, input,
            std::vector<int64_t>{axes.begin(), axes.end()}, op->getKeepDims(),
            std::nullopt, output);
    }
};

class ReduceMaxInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op, const RuntimeObj *context) const override {
        auto op = as<ReduceMaxObj>(_op);
        auto input = infiniops::toInfiniOpsTensor(op->getInputs(0), context);
        auto output = infiniops::toInfiniOpsTensor(op->getOutput(), context);
        auto handle = infiniops::makeInfiniOpsHandle(context);
        auto config = infiniops::makeInfiniOpsConfig<::infini::ops::Amax>(context);
        const auto &axes = op->getAxes();
        infiniops::dispatch::callAmax(handle, config, input,
            {axes.begin(), axes.end()}, op->getKeepDims(), output);
    }
};

} // namespace

REGISTER_KERNEL(ExecutionProvider::Infini, OpType::ReduceL2, ReduceL2InfiniOps,
                "ReduceL2_InfiniOps");

REGISTER_KERNEL(ExecutionProvider::Infini, OpType::ReduceMean,
                ReduceMeanInfiniOps, "ReduceMean_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::ReduceSum,
                ReduceSumInfiniOps, "ReduceSum_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::ReduceMax,
                ReduceMaxInfiniOps, "ReduceMax_InfiniOps");

} // namespace infini

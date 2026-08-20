#include "infiniops_common.h"
#include "infiniops_dispatch.h"
#include "operators/expand.h"
#include "operators/slice.h"
#include "operators/transpose.h"

#ifdef USE_INFINIOPS_ATEN_KERNELS
#include <base/expand_copy.h>
#include <base/permute_copy.h>
#include <base/slice_copy.h>
#endif

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace infini {
namespace {

#ifdef USE_INFINIOPS_ATEN_KERNELS
std::vector<int64_t> toInt64(const Shape &values) {
    std::vector<int64_t> result;
    result.reserve(values.size());
    for (const auto value : values) {
        result.push_back(static_cast<int64_t>(value));
    }
    return result;
}

class ExpandAtenInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<ExpandObj>(_op);
        auto input = infiniops::toInfiniOpsTensor(op->getInputs(0), context);
        auto output = infiniops::toInfiniOpsTensor(op->getOutput(), context);
        auto handle = infiniops::makeInfiniOpsHandle(context);
        auto config =
            infiniops::makeInfiniOpsAtenConfig<::infini::ops::ExpandCopy>(
                context);
        infiniops::dispatch::callExpandCopy(
            handle, config, input, toInt64(op->getShape()), false, output);
    }
};

class TransposeAtenInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<TransposeObj>(_op);
        auto input = infiniops::toInfiniOpsTensor(op->getInputs(0), context);
        auto output = infiniops::toInfiniOpsTensor(op->getOutput(), context);
        auto handle = infiniops::makeInfiniOpsHandle(context);
        auto config =
            infiniops::makeInfiniOpsAtenConfig<::infini::ops::PermuteCopy>(
                context);
        infiniops::dispatch::callPermuteCopy(handle, config, input,
                                             toInt64(op->getPermute()), output);
    }
};

class SliceAtenInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<SliceObj>(_op);
        const auto inputTensor = op->getInputs(0);
        const auto starts = op->getStarts();
        const auto steps = op->getSteps();
        const auto inputStrides = inputTensor->getStride();
        const auto outputDims = op->getOutput()->getDims();
        IT_ASSERT(!outputDims.empty());
        IT_ASSERT(starts.size() == outputDims.size());
        IT_ASSERT(steps.size() == outputDims.size());

        size_t offset = 0;
        ::infini::rt::TensorView::Shape shape(outputDims.size());
        ::infini::rt::TensorView::Strides strides(outputDims.size());
        for (size_t i = 0; i < outputDims.size(); ++i) {
            IT_ASSERT(starts[i] >= 0);
            IT_ASSERT(steps[i] > 0,
                      "ATen Slice bridge does not support negative steps");
            offset += static_cast<size_t>(starts[i]) *
                      static_cast<size_t>(inputStrides[i]);
            shape[i] =
                static_cast<::infini::rt::TensorView::Size>(outputDims[i]);
            strides[i] = static_cast<::infini::rt::TensorView::Stride>(
                inputStrides[i] * steps[i]);
        }

        auto *data = inputTensor->getRawDataPtr<uint8_t *>() +
                     offset * inputTensor->getDType().getSize();
        auto input = infiniops::makeInfiniOpsTensor(
            data, shape,
            infiniops::toInfiniOpsDataType(inputTensor->getDType()), context,
            strides);
        auto output = infiniops::toInfiniOpsTensor(op->getOutput(), context);
        auto handle = infiniops::makeInfiniOpsHandle(context);
        auto config =
            infiniops::makeInfiniOpsAtenConfig<::infini::ops::SliceCopy>(
                context);
        infiniops::dispatch::callSliceCopy(
            handle, config, input, int64_t{0}, std::optional<int64_t>{},
            std::optional<int64_t>{}, int64_t{1}, output);
    }
};
#endif

} // namespace

#ifdef USE_INFINIOPS_ATEN_KERNELS
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Expand, ExpandAtenInfiniOps,
                "Expand_InfiniOps_ATen");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Slice, SliceAtenInfiniOps,
                "Slice_InfiniOps_ATen");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Transpose,
                TransposeAtenInfiniOps, "Transpose_InfiniOps_ATen");
#endif

} // namespace infini

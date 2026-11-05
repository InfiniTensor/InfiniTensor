#include "infiniops_common.h"
#include "infiniops_dispatch.h"
#include "operators/expand.h"
#include "operators/slice.h"
#include "operators/split.h"
#include "operators/transpose.h"

#ifdef USE_INFINIOPS_ATEN_KERNELS
#include <base/expand_copy.h>
#include <base/permute_copy.h>
#include <base/slice_copy.h>
#else
#include <base/copy.h>
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
#else
using RtView = ::infini::rt::TensorView;

void copyView(const Tensor &input, const Tensor &output, size_t offset,
              const RtView::Strides &strides, const RuntimeObj *context) {
    IT_ASSERT(input->getDType() == output->getDType());
    if (output->size() == 0)
        return;
    const auto dtype = input->getDType();
    const auto rtType = infiniops::toInfiniOpsDataType(dtype);
    const auto shape = output->getDims();
    const auto outStrides = output->getStride();
    RtView::Shape viewShape(shape.size());
    RtView::Strides viewOutStrides(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        viewShape[i] = shape[i];
        viewOutStrides[i] = outStrides[i];
    }
    auto src = infiniops::makeInfiniOpsTensor(
        input->getRawDataPtr<uint8_t *>() + offset * dtype.getSize(),
        viewShape, rtType, context, strides);
    auto dst = infiniops::makeInfiniOpsTensor(output->getRawDataPtr<void *>(),
                                            viewShape, rtType, context,
                                            viewOutStrides);
    auto handle = infiniops::makeInfiniOpsHandle(context);
    auto config = infiniops::makeInfiniOpsConfig<::infini::ops::Copy>(context);
    infiniops::dispatch::callCopy(handle, config, src, true, dst);
}

class TransposeNativeInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op, const RuntimeObj *context) const override {
        auto op = as<TransposeObj>(_op);
        const auto input = op->getInputs(0);
        const auto inputStrides = input->getStride();
        const auto permute = op->getPermute();
        RtView::Strides strides(permute.size());
        for (size_t i = 0; i < permute.size(); ++i)
            strides[i] = inputStrides[permute[i]];
        copyView(input, op->getOutput(), 0, strides, context);
    }
};

class SliceNativeInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op, const RuntimeObj *context) const override {
        auto op = as<SliceObj>(_op);
        const auto input = op->getInputs(0);
        const auto output = op->getOutput();
        const auto inputStrides = input->getStride();
        const auto inputDims = input->getDims();
        const auto outputDims = output->getDims();
        const auto starts = op->getStarts();
        const auto steps = op->getSteps();
        IT_ASSERT(starts.size() == outputDims.size() &&
                  steps.size() == outputDims.size() &&
                  inputDims.size() == outputDims.size());
        if (output->size() == 0)
            return;
        size_t offset = 0;
        RtView::Strides strides(outputDims.size());
        for (size_t i = 0; i < outputDims.size(); ++i) {
            IT_ASSERT(steps[i] > 0, "Native Slice requires positive steps");
            IT_ASSERT(starts[i] >= 0 && outputDims[i] > 0 &&
                      starts[i] + int64_t(outputDims[i] - 1) * steps[i] < inputDims[i],
                      "Native Slice exceeds the input bounds");
            offset += static_cast<size_t>(starts[i]) * inputStrides[i];
            strides[i] = static_cast<RtView::Stride>(inputStrides[i]) * steps[i];
        }
        copyView(input, output, offset, strides, context);
    }
};

class ExpandNativeInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op, const RuntimeObj *context) const override {
        auto op = as<ExpandObj>(_op);
        const auto input = op->getInputs(0);
        const auto output = op->getOutput();
        const auto inputDims = input->getDims();
        const auto outputDims = output->getDims();
        const auto inputStrides = input->getStride();
        IT_ASSERT(inputDims.size() <= outputDims.size());
        RtView::Strides strides(outputDims.size(), 0);
        const auto offset = outputDims.size() - inputDims.size();
        for (size_t i = 0; i < inputDims.size(); ++i) {
            IT_ASSERT(inputDims[i] == 1 || inputDims[i] == outputDims[offset + i]);
            if (inputDims[i] != 1)
                strides[offset + i] = inputStrides[i];
        }
        copyView(input, output, 0, strides, context);
    }
};

class SplitNativeInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op, const RuntimeObj *context) const override {
        const auto op = as<SplitObj>(_op);
        const auto input = op->getInputs(0);
        const auto shape = input->getDims();
        const auto rawStrides = input->getStride();
        const auto axis = op->getDim();
        IT_ASSERT(axis >= 0 && static_cast<size_t>(axis) < shape.size());
        RtView::Strides strides(shape.size());
        for (size_t i = 0; i < shape.size(); ++i)
            strides[i] = rawStrides[i];
        size_t consumed = 0;
        for (const auto &output : op->getOutputs()) {
            const auto outShape = output->getDims();
            IT_ASSERT(outShape.size() == shape.size());
            const auto width = static_cast<size_t>(outShape[axis]);
            IT_ASSERT(consumed + width <= static_cast<size_t>(shape[axis]));
            copyView(input, output, consumed * rawStrides[axis], strides, context);
            consumed += width;
        }
        IT_ASSERT(consumed == static_cast<size_t>(shape[axis]));
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
#else
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Transpose,
                TransposeNativeInfiniOps, "Transpose_InfiniOps_Copy");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Slice,
                SliceNativeInfiniOps, "Slice_InfiniOps_Copy");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Expand,
                ExpandNativeInfiniOps, "Expand_InfiniOps_Copy");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Split,
                SplitNativeInfiniOps, "Split_InfiniOps_Copy");
#endif

} // namespace infini

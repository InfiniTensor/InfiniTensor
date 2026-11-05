#include "operators/conv.h"
#include "infiniops_common.h"
#include "infiniops_dispatch.h"

#include <base/convolution.h>
#include <base/relu.h>
#include <base/sigmoid.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace infini {
namespace {

std::vector<int64_t> toI64(std::initializer_list<int> values) {
    std::vector<int64_t> ret;
    ret.reserve(values.size());
    for (auto value : values)
        ret.emplace_back(static_cast<int64_t>(value));
    return ret;
}

void applyConvActivation(ActType act, const ::infini::ops::Tensor &output,
                         const ::infini::ops::Handle &handle,
                         const RuntimeObj *context) {
    switch (act) {
    case ActType::None:
        return;
    case ActType::Relu: {
        auto config =
            infiniops::makeInfiniOpsAtenConfig<::infini::ops::Relu>(context);
        infiniops::dispatch::callRelu(handle, config, output, output);
        return;
    }
    case ActType::Sigmoid: {
        auto config =
            infiniops::makeInfiniOpsAtenConfig<::infini::ops::Sigmoid>(context);
        infiniops::dispatch::callSigmoid(handle, config, output, output);
        return;
    }
    default:
        IT_TODO_HALT_MSG("Unsupported InfiniOps Conv activation");
    }
}

void runConvolution(const Tensor &inputTensor, const Tensor &weightTensor,
                    const Tensor &outputTensor,
                    const std::vector<int64_t> &strides,
                    const std::vector<int64_t> &pads,
                    const std::vector<int64_t> &dilations, int groups,
                    ActType activation, const RuntimeObj *context,
                    bool transposed = false,
                    std::vector<int64_t> outputPadding = {},
                    const Tensor &biasTensor = nullptr,
                    bool channelsLast = false) {
    const auto view = [context, channelsLast](const Tensor &tensor) {
        auto packed = infiniops::toInfiniOpsTensor(tensor, context);
        if (!channelsLast)
            return packed;
        const auto dims = packed.shape();
        const auto strides = packed.strides();
        return infiniops::makeInfiniOpsTensor(
            packed.data(), {dims[0], dims[3], dims[1], dims[2]}, packed.dtype(), context,
            {strides[0], strides[3], strides[1], strides[2]});
    };
    auto input = view(inputTensor);
    auto weight = view(weightTensor);
    auto output = view(outputTensor);
    auto handle = infiniops::makeInfiniOpsHandle(context);

#ifdef USE_INFINIOPS_ATEN_KERNELS
    auto config = infiniops::makeInfiniOpsAtenConfig<::infini::ops::Convolution>(context);
#else
    auto config = infiniops::makeInfiniOpsConfig<::infini::ops::Convolution>(context);
#endif
    if (outputPadding.empty())
        outputPadding.resize(strides.size(), 0);
    std::optional<::infini::ops::Tensor> bias;
    if (biasTensor)
        bias.emplace(infiniops::toInfiniOpsTensor(biasTensor, context));
    infiniops::dispatch::callConvolution(
        handle, config, input, weight, bias,
        strides, pads, dilations, transposed, outputPadding, static_cast<int64_t>(groups),
        output);
    applyConvActivation(activation, output, handle, context);
}

class ConvInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<ConvObj>(_op);
        const auto [ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
        runConvolution(op->getInputs(0), op->getInputs(1), op->getOutput(),
                       toI64({sh, sw}), toI64({ph, pw}), toI64({dh, dw}),
                       op->getNumGroups(), op->getAct(), context, false, {},
                       op->getBias(), op->isChannelsLast());
    }
};

class Conv3dInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<Conv3dObj>(_op);
        IT_ASSERT(op->getInputs().size() == 2);
        const auto [pd, ph, pw, sd, sh, sw, dd, dh, dw] =
            op->getPadStrideDilation();
        runConvolution(op->getInputs(0), op->getInputs(1), op->getOutput(),
                       toI64({sd, sh, sw}), toI64({pd, ph, pw}),
                       toI64({dd, dh, dw}), op->getNumGroups(), op->getAct(),
                       context);
    }
};

class ConvTransposeInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op, const RuntimeObj *context) const override {
        auto op = as<ConvTransposed2dObj>(_op);
        const auto [ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
        const auto [oph, opw] = op->getOutputPadding();
        runConvolution(op->getInputs(0), op->getInputs(1), op->getOutput(),
                       toI64({sh, sw}), toI64({ph, pw}), toI64({dh, dw}),
                       op->getNumGroups(), op->getAct(), context, true, toI64({oph, opw}));
    }
};

} // namespace

REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Conv, ConvInfiniOps,
                "Conv_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Conv3d, Conv3dInfiniOps,
                "Conv3d_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::ConvTranspose, ConvTransposeInfiniOps,
                "ConvTranspose_InfiniOps");

} // namespace infini

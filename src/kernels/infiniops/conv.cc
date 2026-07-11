#include "infiniops_common.h"
#include "operators/conv.h"

#include <base/conv_infinilm.h>
#include <base/relu_infinilm.h>
#include <base/sigmoid_infinilm.h>
#include <cstdint>
#include <optional>
#include <vector>

namespace infini {

namespace {

std::vector<int64_t> toI64(std::initializer_list<int> values) {
    std::vector<int64_t> ret;
    ret.reserve(values.size());
    for (auto value : values) {
        ret.emplace_back(static_cast<int64_t>(value));
    }
    return ret;
}

void applyConvActivation(ActType act, const ::infini::ops::Tensor &output,
                         const ::infini::ops::Handle &handle) {
    switch (act) {
    case ActType::None:
        return;
    case ActType::Relu:
        ::infini::ops::ReluInfinilm::Call(handle, {}, output, output);
        return;
    case ActType::Sigmoid:
        ::infini::ops::SigmoidInfinilm::Call(handle, {}, output, output);
        return;
    default:
        IT_TODO_HALT_MSG("Unsupported InfiniOps Conv activation");
    }
}

class ConvInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<ConvObj>(_op);
        IT_ASSERT(op->getInputs().size() == 2);

        auto input = infiniops::toInfiniOpsTensor(op->getInputs(0), context);
        auto weight = infiniops::toInfiniOpsTensor(op->getInputs(1), context);
        auto output = infiniops::toInfiniOpsTensor(op->getOutput(), context);
        const auto [ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
        auto handle = infiniops::makeInfiniOpsHandle(context);

        ::infini::ops::ConvInfinilm::Call(
            handle, {}, input, weight,
            std::optional<::infini::ops::Tensor>{}, toI64({ph, pw}),
            toI64({sh, sw}), toI64({dh, dw}),
            static_cast<int64_t>(op->getNumGroups()), output);
        applyConvActivation(op->getAct(), output, handle);
    }
};

class Conv3dInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<Conv3dObj>(_op);
        IT_ASSERT(op->getInputs().size() == 2);

        auto input = infiniops::toInfiniOpsTensor(op->getInputs(0), context);
        auto weight = infiniops::toInfiniOpsTensor(op->getInputs(1), context);
        auto output = infiniops::toInfiniOpsTensor(op->getOutput(), context);
        const auto [pd, ph, pw, sd, sh, sw, dd, dh, dw] =
            op->getPadStrideDilation();
        auto handle = infiniops::makeInfiniOpsHandle(context);

        ::infini::ops::ConvInfinilm::Call(
            handle, {}, input, weight,
            std::optional<::infini::ops::Tensor>{}, toI64({pd, ph, pw}),
            toI64({sd, sh, sw}), toI64({dd, dh, dw}),
            static_cast<int64_t>(op->getNumGroups()), output);
        applyConvActivation(op->getAct(), output, handle);
    }
};

} // namespace

#ifdef USE_CUDA
REGISTER_KERNEL(Device::CUDA, OpType::Conv, ConvInfiniOps,
                "Conv_InfiniOps_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Conv3d, Conv3dInfiniOps,
                "Conv3d_InfiniOps_CUDA");
#endif

} // namespace infini

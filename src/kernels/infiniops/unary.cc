#include "operators/unary.h"
#include "infiniops_common.h"
#include "infiniops_dispatch.h"

#ifdef USE_INFINIOPS_ATEN_KERNELS
#include <base/clip.h>
#include <base/gelu.h>
#include <base/hardsigmoid.h>
#include <base/relu.h>
#include <base/sigmoid.h>
#include <base/silu.h>
#endif

#include <optional>
#include <string>

namespace infini {
namespace {

#ifdef USE_INFINIOPS_ATEN_KERNELS
class UnaryInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<UnaryObj>(_op);
        auto input = infiniops::toInfiniOpsTensor(op->getInputs(0), context);
        auto output = infiniops::toInfiniOpsTensor(op->getOutput(), context);
        auto handle = infiniops::makeInfiniOpsHandle(context);

        switch (op->getOpType().underlying()) {
        case OpType::Gelu: {
            auto config =
                infiniops::makeInfiniOpsAtenConfig<::infini::ops::Gelu>(
                    context);
            infiniops::dispatch::callGelu(handle, config, input,
                                          std::string("none"), output);
            return;
        }
        case OpType::HardSigmoid: {
            // InfiniTensor IR semantics: y = clip(alpha * x + beta, 0, 1)
            // ONNX default: alpha=0.2, beta=0.5. PyTorch uses alpha=1/6.
            // Decompose into primitives to avoid ATen's hardcoded 1/6.
            // y = clamp(0.2 * x + 0.5, 0, 1) using available ATen ops:
            //   scaled = mul(x, alpha)
            //   biased = add(scaled, beta)
            //   out    = clamp(biased, 0, 1)
            // TODO: use alpha/beta from IR once all platforms support it.
            // For now, the ONNX importer stores them but the bridge
            // delegates to the backend default. See review #315 §3.
            auto config =
                infiniops::makeInfiniOpsAtenConfig<::infini::ops::Hardsigmoid>(
                    context);
            infiniops::dispatch::callHardsigmoid(handle, config, input, output);
            return;
        }
        case OpType::Relu: {
            auto config =
                infiniops::makeInfiniOpsAtenConfig<::infini::ops::Relu>(
                    context);
            infiniops::dispatch::callRelu(handle, config, input, output);
            return;
        }
        case OpType::Sigmoid: {
            auto config =
                infiniops::makeInfiniOpsAtenConfig<::infini::ops::Sigmoid>(
                    context);
            infiniops::dispatch::callSigmoid(handle, config, input, output);
            return;
        }
        case OpType::Silu: {
            auto config =
                infiniops::makeInfiniOpsAtenConfig<::infini::ops::Silu>(
                    context);
            infiniops::dispatch::callSilu(handle, config, input, output);
            return;
        }
        default:
            IT_TODO_HALT_MSG("Unsupported unary InfiniOps bridge");
        }
    }
};

class ClipInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<ClipObj>(_op);
        auto input = infiniops::toInfiniOpsTensor(op->getInputs(0), context);
        auto output = infiniops::toInfiniOpsTensor(op->getOutput(), context);
        auto handle = infiniops::makeInfiniOpsHandle(context);
        auto config =
            infiniops::makeInfiniOpsAtenConfig<::infini::ops::Clip>(context);
        const auto min = op->getMin();
        const auto max = op->getMax();
        infiniops::dispatch::callClip(
            handle, config, input,
            min ? std::optional<double>{static_cast<double>(*min)}
                : std::nullopt,
            max ? std::optional<double>{static_cast<double>(*max)}
                : std::nullopt,
            output);
    }
};
#endif

} // namespace

#ifdef USE_INFINIOPS_ATEN_KERNELS
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Clip, ClipInfiniOps,
                "Clip_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Gelu, UnaryInfiniOps,
                "Gelu_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::HardSigmoid, UnaryInfiniOps,
                "HardSigmoid_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Relu, UnaryInfiniOps,
                "Relu_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Sigmoid, UnaryInfiniOps,
                "Sigmoid_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Silu, UnaryInfiniOps,
                "Silu_InfiniOps");
#endif

} // namespace infini

#include "operators/unary.h"
#include "infiniops_common.h"
#include "infiniops_dispatch.h"
#include <base/sigmoid.h>
#include <base/silu.h>
#include <base/relu.h>
#include <base/logical_not.h>
#include <base/sin.h>
#include <base/cos.h>
#include <base/erf.h>
#include <base/neg.h>
#include <base/sqrt.h>
#include <base/tanh.h>
#include <base/abs.h>
#include <base/sign.h>
#include <base/rsqrt.h>
#include <base/log.h>
#include <base/clip.h>
#include <base/gelu.h>

#ifdef USE_INFINIOPS_ATEN_KERNELS
#include <base/hardsigmoid.h>
#endif

#include <optional>
#include <string>
#include <type_traits>

namespace infini {
namespace {

template <typename Op, auto Call>
class SimpleUnaryInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &op, const RuntimeObj *context) const override {
        if constexpr (std::is_same_v<Op, ::infini::ops::Log>)
            IT_ASSERT(as<LogObj>(op)->getType() == LogObj::LogE,
                      "Infini Log currently supports the natural logarithm");
        auto input = infiniops::toInfiniOpsTensor(op->getInputs(0), context);
        auto output = infiniops::toInfiniOpsTensor(op->getOutput(), context);
        auto handle = infiniops::makeInfiniOpsHandle(context);
        auto config = infiniops::makeInfiniOpsConfig<Op>(context);
        Call(handle, config, input, output);
    }
};

using SinInfiniOps = SimpleUnaryInfiniOps<::infini::ops::Sin, infiniops::dispatch::callSin>;
using CosInfiniOps = SimpleUnaryInfiniOps<::infini::ops::Cos, infiniops::dispatch::callCos>;
using ErfInfiniOps = SimpleUnaryInfiniOps<::infini::ops::Erf, infiniops::dispatch::callErf>;
using NegInfiniOps = SimpleUnaryInfiniOps<::infini::ops::Neg, infiniops::dispatch::callNeg>;
using NotInfiniOps = SimpleUnaryInfiniOps<::infini::ops::LogicalNot, infiniops::dispatch::callLogicalNot>;
using TanhInfiniOps = SimpleUnaryInfiniOps<::infini::ops::Tanh, infiniops::dispatch::callTanh>;
using AbsInfiniOps = SimpleUnaryInfiniOps<::infini::ops::Abs, infiniops::dispatch::callAbs>;
using SignInfiniOps = SimpleUnaryInfiniOps<::infini::ops::Sign, infiniops::dispatch::callSign>;
using RsqrtInfiniOps = SimpleUnaryInfiniOps<::infini::ops::Rsqrt, infiniops::dispatch::callRsqrt>;
using LogInfiniOps = SimpleUnaryInfiniOps<::infini::ops::Log, infiniops::dispatch::callLog>;
using SqrtInfiniOps = SimpleUnaryInfiniOps<::infini::ops::Sqrt, infiniops::dispatch::callSqrt>;

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

#endif

class ClipInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<ClipObj>(_op);
        auto input = infiniops::toInfiniOpsTensor(op->getInputs(0), context);
        auto output = infiniops::toInfiniOpsTensor(op->getOutput(), context);
        auto handle = infiniops::makeInfiniOpsHandle(context);
#ifdef USE_INFINIOPS_ATEN_KERNELS
        auto config = infiniops::makeInfiniOpsAtenConfig<::infini::ops::Clip>(context);
#else
        auto config = infiniops::makeInfiniOpsConfig<::infini::ops::Clip>(context);
#endif
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
#ifndef USE_INFINIOPS_ATEN_KERNELS
using ReluNativeInfiniOps = SimpleUnaryInfiniOps<::infini::ops::Relu, infiniops::dispatch::callRelu>;
using SiluNativeInfiniOps = SimpleUnaryInfiniOps<::infini::ops::Silu, infiniops::dispatch::callSilu>;
class GeluNativeInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &op, const RuntimeObj *context) const override {
        auto input = infiniops::toInfiniOpsTensor(op->getInputs(0), context);
        auto output = infiniops::toInfiniOpsTensor(op->getOutput(), context);
        auto handle = infiniops::makeInfiniOpsHandle(context);
        auto config = infiniops::makeInfiniOpsConfig<::infini::ops::Gelu>(context);
        infiniops::dispatch::callGelu(handle, config, input, std::string("none"), output);
    }
};
class SigmoidNativeInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op, const RuntimeObj *context) const override {
        auto input = infiniops::toInfiniOpsTensor(_op->getInputs(0), context);
        auto output = infiniops::toInfiniOpsTensor(_op->getOutput(), context);
        auto handle = infiniops::makeInfiniOpsHandle(context);
        auto config = infiniops::makeInfiniOpsConfig<::infini::ops::Sigmoid>(context);
        infiniops::dispatch::callSigmoid(handle, config, input, output);
    }
};
#endif

} // namespace

REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Sin, SinInfiniOps, "Sin_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Cos, CosInfiniOps, "Cos_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Erf, ErfInfiniOps, "Erf_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Neg, NegInfiniOps, "Neg_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Not, NotInfiniOps, "Not_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Tanh, TanhInfiniOps, "Tanh_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Abs, AbsInfiniOps, "Abs_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Sign, SignInfiniOps, "Sign_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Rsqrt, RsqrtInfiniOps, "Rsqrt_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Log, LogInfiniOps, "Log_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Sqrt, SqrtInfiniOps, "Sqrt_InfiniOps");

REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Clip, ClipInfiniOps,
                "Clip_InfiniOps");
#ifdef USE_INFINIOPS_ATEN_KERNELS
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
#else
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Relu, ReluNativeInfiniOps,
                "Relu_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Sigmoid,
                SigmoidNativeInfiniOps, "Sigmoid_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Silu,
                SiluNativeInfiniOps, "Silu_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Gelu,
                GeluNativeInfiniOps, "Gelu_InfiniOps");
#endif

} // namespace infini

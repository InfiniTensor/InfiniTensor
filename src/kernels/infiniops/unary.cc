#include "infiniops_common.h"
#include "operators/unary.h"

#include <base/gelu_infinilm.h>
#include <base/relu_infinilm.h>
#include <base/sigmoid_infinilm.h>
#include <base/silu.h>
#include <string>

namespace infini {
namespace {

class UnaryInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<UnaryObj>(_op);

        auto input = infiniops::toInfiniOpsTensor(op->getInputs(0), context);
        auto output = infiniops::toInfiniOpsTensor(op->getOutput(), context);
        auto handle = infiniops::makeInfiniOpsHandle(context);

        switch (op->getOpType().underlying()) {
        case OpType::Gelu:
            ::infini::ops::GeluInfinilm::Call(handle, {}, input,
                                              std::string("none"), output);
            return;
        case OpType::Relu:
            ::infini::ops::ReluInfinilm::Call(handle, {}, input, output);
            return;
        case OpType::Sigmoid:
            ::infini::ops::SigmoidInfinilm::Call(handle, {}, input, output);
            return;
        case OpType::Silu:
            ::infini::ops::Silu::Call(handle, {}, input, output);
            return;
        default:
            IT_TODO_HALT();
        }
    }
};

} // namespace

#ifdef USE_CUDA
REGISTER_KERNEL(Device::CUDA, OpType::Gelu, UnaryInfiniOps,
                "Gelu_InfiniOps_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Relu, UnaryInfiniOps,
                "Relu_InfiniOps_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Sigmoid, UnaryInfiniOps,
                "Sigmoid_InfiniOps_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Silu, UnaryInfiniOps,
                "Silu_InfiniOps_CUDA");
#endif
#ifdef USE_ILUVATAR
REGISTER_KERNEL(Device::ILUVATAR, OpType::Gelu, UnaryInfiniOps, "Gelu_InfiniOps_ILUVATAR");
REGISTER_KERNEL(Device::ILUVATAR, OpType::Relu, UnaryInfiniOps, "Relu_InfiniOps_ILUVATAR");
REGISTER_KERNEL(Device::ILUVATAR, OpType::Sigmoid, UnaryInfiniOps, "Sigmoid_InfiniOps_ILUVATAR");
REGISTER_KERNEL(Device::ILUVATAR, OpType::Silu, UnaryInfiniOps, "Silu_InfiniOps_ILUVATAR");
#endif
#ifdef USE_METAX
REGISTER_KERNEL(Device::METAX, OpType::Gelu, UnaryInfiniOps, "Gelu_InfiniOps_METAX");
REGISTER_KERNEL(Device::METAX, OpType::Relu, UnaryInfiniOps, "Relu_InfiniOps_METAX");
REGISTER_KERNEL(Device::METAX, OpType::Sigmoid, UnaryInfiniOps, "Sigmoid_InfiniOps_METAX");
REGISTER_KERNEL(Device::METAX, OpType::Silu, UnaryInfiniOps, "Silu_InfiniOps_METAX");
#endif
#ifdef USE_MOORE
REGISTER_KERNEL(Device::MOORE, OpType::Gelu, UnaryInfiniOps, "Gelu_InfiniOps_MOORE");
REGISTER_KERNEL(Device::MOORE, OpType::Relu, UnaryInfiniOps, "Relu_InfiniOps_MOORE");
REGISTER_KERNEL(Device::MOORE, OpType::Sigmoid, UnaryInfiniOps, "Sigmoid_InfiniOps_MOORE");
REGISTER_KERNEL(Device::MOORE, OpType::Silu, UnaryInfiniOps, "Silu_InfiniOps_MOORE");
#endif

} // namespace infini

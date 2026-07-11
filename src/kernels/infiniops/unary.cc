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

} // namespace infini

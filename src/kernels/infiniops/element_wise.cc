#include "operators/element_wise.h"
#include "infiniops_common.h"
#include "infiniops_dispatch.h"

#include <base/add.h>
#include <base/mul.h>

namespace infini {
namespace {

class AddInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<ElementWiseObj>(_op);

        auto outputDims = op->getOutput()->getDims();
        auto input0 = infiniops::toInfiniOpsBroadcastTensor(
            op->getInputs(0), outputDims, context);
        auto input1 = infiniops::toInfiniOpsBroadcastTensor(
            op->getInputs(1), outputDims, context);
        auto output = infiniops::toInfiniOpsTensor(op->getOutput(), context);
        auto handle = infiniops::makeInfiniOpsHandle(context);

        auto config =
            infiniops::makeInfiniOpsConfig<::infini::ops::Add>(context);
        infiniops::dispatch::callAdd(handle, config, input0, input1, 1.0,
                                     output);
    }
};

class MulInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<ElementWiseObj>(_op);
        const auto outputDims = op->getOutput()->getDims();
        auto input0 = infiniops::toInfiniOpsBroadcastTensor(
            op->getInputs(0), outputDims, context);
        auto input1 = infiniops::toInfiniOpsBroadcastTensor(
            op->getInputs(1), outputDims, context);
        auto output = infiniops::toInfiniOpsTensor(op->getOutput(), context);
        auto handle = infiniops::makeInfiniOpsHandle(context);

        auto config =
            infiniops::makeInfiniOpsConfig<::infini::ops::Mul>(context);
        infiniops::dispatch::callMul(handle, config, input0, input1, output);
    }
};

} // namespace

REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Add, AddInfiniOps,
                "Add_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Mul, MulInfiniOps,
                "Mul_InfiniOps");

} // namespace infini

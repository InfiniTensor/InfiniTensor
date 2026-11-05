#include "operators/element_wise.h"
#include "infiniops_common.h"
#include "infiniops_dispatch.h"

#include <base/add.h>
#include <base/div.h>
#include <base/eq.h>
#include <base/ge.h>
#include <base/less.h>
#include <base/mul.h>
#include <base/pow.h>
#include <base/maximum.h>

namespace infini {
namespace {

template <int Alpha>
class ScaledAddInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<ElementWiseObj>(_op);

#ifdef USE_INFINIOPS_ATEN_KERNELS
        auto outputDims = op->getOutput()->getDims();
        auto input0 = infiniops::toInfiniOpsBroadcastTensor(
            op->getInputs(0), outputDims, context);
        auto input1 = infiniops::toInfiniOpsBroadcastTensor(
            op->getInputs(1), outputDims, context);
#else
        auto input0 = infiniops::toInfiniOpsTensor(op->getInputs(0), context);
        auto input1 = infiniops::toInfiniOpsTensor(op->getInputs(1), context);
#endif
        auto output = infiniops::toInfiniOpsTensor(op->getOutput(), context);
        auto handle = infiniops::makeInfiniOpsHandle(context);

        auto config =
            infiniops::makeInfiniOpsConfig<::infini::ops::Add>(context);
        infiniops::dispatch::callAdd(handle, config, input0, input1, Alpha,
                                     output);
    }
};

class MulInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<ElementWiseObj>(_op);
#ifdef USE_INFINIOPS_ATEN_KERNELS
        const auto outputDims = op->getOutput()->getDims();
        auto input0 = infiniops::toInfiniOpsBroadcastTensor(
            op->getInputs(0), outputDims, context);
        auto input1 = infiniops::toInfiniOpsBroadcastTensor(
            op->getInputs(1), outputDims, context);
#else
        auto input0 = infiniops::toInfiniOpsTensor(op->getInputs(0), context);
        auto input1 = infiniops::toInfiniOpsTensor(op->getInputs(1), context);
#endif
        auto output = infiniops::toInfiniOpsTensor(op->getOutput(), context);
        auto handle = infiniops::makeInfiniOpsHandle(context);

        auto config =
            infiniops::makeInfiniOpsConfig<::infini::ops::Mul>(context);
        infiniops::dispatch::callMul(handle, config, input0, input1, output);
    }
};

template <typename Key, auto Call>
class BinaryInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<ElementWiseObj>(_op);
        auto input0 = infiniops::toInfiniOpsTensor(op->getInputs(0), context);
        auto input1 = infiniops::toInfiniOpsTensor(op->getInputs(1), context);
        auto output = infiniops::toInfiniOpsTensor(op->getOutput(), context);
        auto handle = infiniops::makeInfiniOpsHandle(context);
        auto config = infiniops::makeInfiniOpsConfig<Key>(context);
        Call(handle, config, input0, input1, output);
    }
};

using DivInfiniOps = BinaryInfiniOps<::infini::ops::Div, infiniops::dispatch::callDiv>;
using EqualInfiniOps = BinaryInfiniOps<::infini::ops::Eq, infiniops::dispatch::callEq>;
using GreaterEqualInfiniOps = BinaryInfiniOps<::infini::ops::Ge, infiniops::dispatch::callGe>;
using PowInfiniOps = BinaryInfiniOps<::infini::ops::Pow, infiniops::dispatch::callPow>;
using MaximumInfiniOps = BinaryInfiniOps<::infini::ops::Maximum, infiniops::dispatch::callMaximum>;
using LessInfiniOps = BinaryInfiniOps<::infini::ops::Less, infiniops::dispatch::callLess>;

} // namespace

REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Add, ScaledAddInfiniOps<1>,
                "Add_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Sub, ScaledAddInfiniOps<-1>,
                "Sub_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Mul, MulInfiniOps,
                "Mul_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Div, DivInfiniOps,
                "Div_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Equal, EqualInfiniOps, "Equal_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::GreaterOrEqual, GreaterEqualInfiniOps,
                "GreaterEqual_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Pow, PowInfiniOps, "Pow_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Max, MaximumInfiniOps, "Maximum_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Less, LessInfiniOps, "Less_InfiniOps");

} // namespace infini

#include "infiniops_common.h"
#include "infiniops_dispatch.h"
#include "operators/where.h"

#include <base/where.h>

namespace infini {
namespace {

class WhereInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op, const RuntimeObj *context) const override {
        auto op = as<WhereObj>(_op);
        IT_ASSERT(op->getInputs(2)->getDType() == DataType::Bool);
        auto condition = infiniops::toInfiniOpsTensor(op->getInputs(2), context);
        auto input = infiniops::toInfiniOpsTensor(op->getInputs(0), context);
        auto other = infiniops::toInfiniOpsTensor(op->getInputs(1), context);
        auto output = infiniops::toInfiniOpsTensor(op->getOutput(), context);
        auto handle = infiniops::makeInfiniOpsHandle(context);
        auto config = infiniops::makeInfiniOpsConfig<::infini::ops::Where>(context);
        infiniops::dispatch::callWhere(handle, config, condition, input, other,
                                      output);
    }
};

} // namespace

REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Where, WhereInfiniOps,
                "Where_InfiniOps");
} // namespace infini

#include "operators/softmax.h"
#include "infiniops_common.h"
#include "infiniops_dispatch.h"

#include <base/softmax.h>

#include <optional>

namespace infini {
namespace {

class SoftmaxInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<SoftmaxObj>(_op);
        auto input = infiniops::toInfiniOpsTensor(op->getInputs(0), context);
        auto output = infiniops::toInfiniOpsTensor(op->getOutput(), context);
        auto handle = infiniops::makeInfiniOpsHandle(context);
#ifdef USE_INFINIOPS_ATEN_KERNELS
        auto config =
            infiniops::makeInfiniOpsAtenConfig<::infini::ops::Softmax>(context);
#else
        auto config =
            infiniops::makeInfiniOpsConfig<::infini::ops::Softmax>(context);
#endif
        infiniops::dispatch::callSoftmax(
            handle, config, input, static_cast<int64_t>(op->getAxis()),
            std::optional<::infini::rt::DataType>{}, output);
    }
};

} // namespace

REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Softmax, SoftmaxInfiniOps,
                "Softmax_InfiniOps");

} // namespace infini

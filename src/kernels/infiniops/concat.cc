#include "operators/concat.h"
#include "infiniops_common.h"
#include "infiniops_dispatch.h"

#include <base/cat.h>
#include <cstdint>
#include <vector>

namespace infini {
namespace {

class ConcatInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<ConcatObj>(_op);
        std::vector<::infini::ops::Tensor> inputs;
        inputs.reserve(op->getInputs().size());
        for (const auto &input : op->getInputs()) {
            inputs.push_back(infiniops::toInfiniOpsTensor(input, context));
        }

        auto output = infiniops::toInfiniOpsTensor(op->getOutput(), context);
        auto handle = infiniops::makeInfiniOpsHandle(context);
#ifdef USE_INFINIOPS_ATEN_KERNELS
        auto config = infiniops::makeInfiniOpsAtenConfig<::infini::ops::Cat>(context);
#else
        auto config = infiniops::makeInfiniOpsConfig<::infini::ops::Cat>(context);
#endif
        infiniops::dispatch::callCat(
            handle, config, inputs, static_cast<int64_t>(op->getDim()), output);
    }
};

} // namespace

REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Concat, ConcatInfiniOps,
                "Concat_InfiniOps");

} // namespace infini

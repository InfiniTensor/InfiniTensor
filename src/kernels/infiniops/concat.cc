#include "operators/concat.h"
#include "infiniops_common.h"
#include "infiniops_dispatch.h"

#ifdef USE_INFINIOPS_ATEN_KERNELS
#include <base/cat.h>
#endif
#include <cstdint>
#include <vector>

namespace infini {
namespace {

#ifdef USE_INFINIOPS_ATEN_KERNELS
class ConcatAtenInfiniOps : public infiniops::KernelWithoutConfig {
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
        auto config =
            infiniops::makeInfiniOpsAtenConfig<::infini::ops::Cat>(context);
        infiniops::dispatch::callCat(
            handle, config, inputs, static_cast<int64_t>(op->getDim()), output);
    }
};
#endif

} // namespace

#ifdef USE_INFINIOPS_ATEN_KERNELS
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Concat, ConcatAtenInfiniOps,
                "Concat_InfiniOps");
#endif

} // namespace infini

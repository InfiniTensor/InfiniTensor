#include "operators/concat.h"
#include "core/data_type.h"
#include "core/kernel.h"
#include "core/tensor.h"
#include "cpu/cat/cat.h"
#ifdef WITH_ASCEND
#include "ascend/cat/kernel.h"
#endif

namespace infini {

class ConcatInfiniOpsKernel : public KernelWithoutConfig {
    void compute(const Operator &op, const RuntimeObj *context) const override {
        auto concatOp = as<ConcatObj>(op);

        auto inputs = concatOp->getInputs();
        IT_ASSERT(inputs.size() >= 2, "Concat requires at least 2 inputs");

        auto first_input = toInfiniOpsTensor(inputs[0].get());
        std::vector<infini::ops::Tensor> rest_inputs;
        for (size_t i = 1; i < inputs.size(); ++i) {
            rest_inputs.push_back(toInfiniOpsTensor(inputs[i].get()));
        }
        auto output = toInfiniOpsTensor(concatOp->getOutput().get());

        infini::ops::Handle handle = context->makeHandle();
        infini::ops::Config config;
        config.set_implementation_index(
            context->resolveImplementationIndex<infini::ops::Cat>());

        infini::ops::Cat::Call(handle, config, first_input, rest_inputs,
                               concatOp->getDim(), output);
    }
};

// Cat only has CPU and Ascend implementations in InfiniOps.
REGISTER_KERNEL(Device(Device::Type::kCpu), OpType::Concat,
                ConcatInfiniOpsKernel, "Concat_InfiniOps_CPU");
REGISTER_KERNEL(Device(Device::Type::kAscend), OpType::Concat,
                ConcatInfiniOpsKernel, "Concat_InfiniOps_ASCEND");

} // namespace infini

#include "core/infiniops_bridge/adapter_kernel.h"
#include "core/infiniops_bridge/tensor_convert.h"
#include "cpu/cat/cat.h"
#include "operators/concat.h"

namespace infini {

class ConcatInfiniOpsKernel : public InfiniOpsAdapterKernel {
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

        infini::ops::Handle handle;
        infini::ops::Config config;

        infini::ops::Cat::Call(handle, config, first_input, rest_inputs,
                                 concatOp->getDim(), output);
    }
};

REGISTER_KERNEL(Device(Device::Type::kCpu), OpType::Concat, ConcatInfiniOpsKernel,
                "Concat_InfiniOps_CPU");

} // namespace infini

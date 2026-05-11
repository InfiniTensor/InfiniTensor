#include "cpu/add/add.h"
#ifdef WITH_NVIDIA
#include "cuda/nvidia/add/kernel.h"
#endif
#ifdef WITH_CAMBRICON
// TODO: add cambricon/add when available
#endif
#ifdef WITH_ASCEND
#include "ascend/add/kernel.h"
#endif
#ifdef WITH_ILUVATAR
#include "cuda/iluvatar/add/kernel.h"
#endif
#ifdef WITH_METAX
#include "cuda/metax/add/kernel.h"
#endif
#ifdef WITH_MOORE
#include "cuda/moore/add/kernel.h"
#endif
#ifdef WITH_TORCH
#include "torch/add/add.h"
#endif
#include "core/data_type.h"
#include "core/kernel.h"
#include "core/tensor.h"
#include "operators/element_wise.h"

namespace infini {

class AddInfiniOpsKernel : public KernelWithoutConfig {
    void compute(const Operator &op, const RuntimeObj *context) const override {
        auto elemOp = as<ElementWiseObj>(op);

        auto input = toInfiniOpsTensor(elemOp->getInputs(0).get());
        auto other = toInfiniOpsTensor(elemOp->getInputs(1).get());
        auto output = toInfiniOpsTensor(elemOp->getOutput().get());

        infini::ops::Handle handle = context->makeHandle();
        infini::ops::Config config;
        config.set_implementation_index(
            context->resolveImplementationIndex<infini::ops::Add>());

        infini::ops::Add::Call(handle, config, input, other, output);
    }
};

REGISTER_ALL_DEVICES(OpType::Add, AddInfiniOpsKernel, "Add_InfiniOps");

} // namespace infini

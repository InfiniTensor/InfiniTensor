#include "native/cpu/ops/add/add.h"
#ifdef WITH_NVIDIA
#include "native/cuda/nvidia/ops/add/kernel.h"
#endif
#ifdef WITH_CAMBRICON
// TODO: add cambricon/add when available
#endif
#ifdef WITH_ASCEND
#include "native/ascend/ops/add/kernel.h"
#endif
#ifdef WITH_ILUVATAR
#include "native/cuda/iluvatar/ops/add/kernel.h"
#endif
#ifdef WITH_METAX
#include "native/cuda/metax/ops/add/kernel.h"
#endif
#ifdef WITH_MOORE
#include "native/cuda/moore/ops/add/kernel.h"
#endif
#ifdef WITH_TORCH
#include "torch/ops/add/add.h"
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

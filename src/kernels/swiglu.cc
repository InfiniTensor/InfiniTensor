#include "cpu/swiglu/swiglu.h"
#ifdef WITH_NVIDIA
#include "cuda/nvidia/swiglu/kernel.h"
#endif
#ifdef WITH_ILUVATAR
#include "cuda/iluvatar/swiglu/kernel.h"
#endif
#ifdef WITH_METAX
#include "cuda/metax/swiglu/kernel.h"
#endif
#ifdef WITH_MOORE
#include "cuda/moore/swiglu/kernel.h"
#endif
#include "core/kernel.h"
#include "core/tensor.h"
#include "operators/swiglu.h"

namespace infini {

class SwiGLUInfiniOpsKernel : public KernelWithoutConfig {
    void compute(const Operator &op, const RuntimeObj *context) const override {
        auto swigluOp = as<SwiGLUObj>(op);

        auto input = toInfiniOpsTensor(swigluOp->getInputs(0).get());
        auto gate = toInfiniOpsTensor(swigluOp->getInputs(1).get());
        auto output = toInfiniOpsTensor(swigluOp->getOutput().get());

        infini::ops::Handle handle = context->makeHandle();
        infini::ops::Config config;
        config.set_implementation_index(
            context->resolveImplementationIndex<infini::ops::Swiglu>());

        infini::ops::Swiglu::Call(handle, config, input, gate, output);
    }
};

REGISTER_ALL_DEVICES(OpType::SwiGLU, SwiGLUInfiniOpsKernel, "SwiGLU_InfiniOps");

} // namespace infini

#include "cpu/rms_norm/rms_norm.h"
#ifdef WITH_NVIDIA
#include "cuda/nvidia/rms_norm/kernel.h"
#endif
#ifdef WITH_CAMBRICON
#include "cambricon/rms_norm/rms_norm.h"
#endif
#ifdef WITH_ILUVATAR
#include "cuda/iluvatar/rms_norm/kernel.h"
#endif
#ifdef WITH_METAX
#include "cuda/metax/rms_norm/kernel.h"
#endif
#ifdef WITH_MOORE
#include "cuda/moore/rms_norm/kernel.h"
#endif
#include "core/data_type.h"
#include "core/kernel.h"
#include "core/tensor.h"
#include "operators/rms_norm.h"

namespace infini {

class RMSNormInfiniOpsKernel : public KernelWithoutConfig {
    void compute(const Operator &op, const RuntimeObj *context) const override {
        auto normOp = as<RMSNormObj>(op);

        auto input = toInfiniOpsTensor(normOp->getInputs(0).get());
        auto weight = toInfiniOpsTensor(normOp->getInputs(1).get());
        auto output = toInfiniOpsTensor(normOp->getOutput().get());

        infini::ops::Handle handle = context->makeHandle();
        infini::ops::Config config;

        // Use default eps (1e-6) — InfiniTensor's RMSNormObj doesn't expose eps
        infini::ops::RmsNorm::Call(handle, config, input, weight, output);
    }
};

REGISTER_ALL_DEVICES(OpType::RMSNorm, RMSNormInfiniOpsKernel,
                     "RMSNorm_InfiniOps");

} // namespace infini

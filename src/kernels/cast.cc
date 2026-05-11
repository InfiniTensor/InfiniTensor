#include "cpu/cast/cast.h"
#ifdef WITH_ASCEND
#include "ascend/cast/kernel.h"
#endif
#include "core/data_type.h"
#include "core/kernel.h"
#include "core/tensor.h"
#include "operators/unary.h"

namespace infini {

class CastInfiniOpsKernel : public KernelWithoutConfig {
    void compute(const Operator &op, const RuntimeObj *context) const override {
        auto castOp = as<CastObj>(op);

        auto input = toInfiniOpsTensor(castOp->getInputs(0).get());
        auto output = toInfiniOpsTensor(castOp->getOutput().get());

        infini::ops::Handle handle = context->makeHandle();
        infini::ops::Config config;
        config.set_implementation_index(
            context->resolveImplementationIndex<infini::ops::Cast>());

        infini::ops::Cast::Call(handle, config, input, output);
    }
};

// Cast only has CPU and Ascend implementations in InfiniOps.
REGISTER_KERNEL(Device(Device::Type::kCpu), OpType::Cast, CastInfiniOpsKernel,
                "Cast_InfiniOps_CPU");
REGISTER_KERNEL(Device(Device::Type::kAscend), OpType::Cast,
                CastInfiniOpsKernel, "Cast_InfiniOps_ASCEND");

} // namespace infini

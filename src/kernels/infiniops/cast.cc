#include "core/kernel.h"
#include "core/data_type.h"
#include "core/tensor.h"
#include "cpu/cast/cast.h"
#include "operators/unary.h"

namespace infini {

class CastInfiniOpsKernel : public KernelWithoutConfig {
    void compute(const Operator &op, const RuntimeObj *context) const override {
        auto castOp = as<CastObj>(op);

        auto input = toInfiniOpsTensor(castOp->getInputs(0).get());
        auto output = toInfiniOpsTensor(castOp->getOutput().get());

        infini::ops::Handle handle = context->makeHandle();
        infini::ops::Config config;

        infini::ops::Cast::Call(handle, config, input, output);
    }
};

REGISTER_KERNEL(Device(Device::Type::kCpu), OpType::Cast, CastInfiniOpsKernel,
                "Cast_InfiniOps_CPU");

} // namespace infini

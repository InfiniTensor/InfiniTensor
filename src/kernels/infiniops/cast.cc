#include "core/infiniops_bridge/adapter_kernel.h"
#include "core/infiniops_bridge/tensor_convert.h"
#include "cpu/cast/cast.h"
#include "operators/unary.h"

namespace infini {

class CastInfiniOpsKernel : public InfiniOpsAdapterKernel {
    void compute(const Operator &op, const RuntimeObj *context) const override {
        auto castOp = as<CastObj>(op);

        auto input = toInfiniOpsTensor(castOp->getInputs(0).get());
        auto output = toInfiniOpsTensor(castOp->getOutput().get());

        infini::ops::Handle handle;
        infini::ops::Config config;

        infini::ops::Cast::Call(handle, config, input, output);
    }
};

REGISTER_KERNEL(Device(Device::Type::kCpu), OpType::Cast, CastInfiniOpsKernel,
                "Cast_InfiniOps_CPU");

} // namespace infini

#include "core/infiniops_bridge/adapter_kernel.h"
#include "core/infiniops_bridge/tensor_convert.h"
#include "cpu/mul/mul.h"
#include "operators/element_wise.h"

namespace infini {

class MulInfiniOpsKernel : public InfiniOpsAdapterKernel {
    void compute(const Operator &op, const RuntimeObj *context) const override {
        auto elemOp = as<ElementWiseObj>(op);

        auto input = toInfiniOpsTensor(elemOp->getInputs(0).get());
        auto other = toInfiniOpsTensor(elemOp->getInputs(1).get());
        auto output = toInfiniOpsTensor(elemOp->getOutput().get());

        infini::ops::Handle handle;
        infini::ops::Config config;

        infini::ops::Mul::Call(handle, config, input, other, output);
    }
};

REGISTER_KERNEL(Device(Device::Type::kCpu), OpType::Mul, MulInfiniOpsKernel,
                "Mul_InfiniOps_CPU");

} // namespace infini

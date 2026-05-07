#include "cpu/mul/mul.h"
#ifdef WITH_ASCEND
#include "ascend/mul/kernel.h"
#endif
#include "core/data_type.h"
#include "core/kernel.h"
#include "core/tensor.h"
#include "operators/element_wise.h"

namespace infini {

class MulInfiniOpsKernel : public KernelWithoutConfig {
    void compute(const Operator &op, const RuntimeObj *context) const override {
        auto elemOp = as<ElementWiseObj>(op);

        auto input = toInfiniOpsTensor(elemOp->getInputs(0).get());
        auto other = toInfiniOpsTensor(elemOp->getInputs(1).get());
        auto output = toInfiniOpsTensor(elemOp->getOutput().get());

        infini::ops::Handle handle = context->makeHandle();
        infini::ops::Config config;

        infini::ops::Mul::Call(handle, config, input, other, output);
    }
};

// Mul only has CPU and Ascend implementations in InfiniOps.
REGISTER_KERNEL(Device(Device::Type::kCpu), OpType::Mul, MulInfiniOpsKernel,
                "Mul_InfiniOps_CPU");
REGISTER_KERNEL(Device(Device::Type::kAscend), OpType::Mul, MulInfiniOpsKernel,
                "Mul_InfiniOps_ASCEND");

} // namespace infini

#include "infiniops_common.h"
#include "operators/element_wise.h"

#include <base/add.h>

namespace infini {
namespace {

class AddInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<ElementWiseObj>(_op);

        auto outputDims = op->getOutput()->getDims();
        auto input0 = infiniops::toInfiniOpsBroadcastTensor(
            op->getInputs(0), outputDims, context);
        auto input1 = infiniops::toInfiniOpsBroadcastTensor(
            op->getInputs(1), outputDims, context);
        auto output = infiniops::toInfiniOpsTensor(op->getOutput(), context);
        auto handle = infiniops::makeInfiniOpsHandle(context);

        ::infini::ops::Add::Call(handle, {}, input0, input1, output);
    }
};

} // namespace

#ifdef USE_CUDA
REGISTER_KERNEL(Device::CUDA, OpType::Add, AddInfiniOps,
                "Add_InfiniOps_CUDA");
#endif
#ifdef USE_ASCEND
REGISTER_KERNEL(Device::ASCEND, OpType::Add, AddInfiniOps,
                "Add_InfiniOps_ASCEND");
#endif
#ifdef USE_ILUVATAR
REGISTER_KERNEL(Device::ILUVATAR, OpType::Add, AddInfiniOps,
                "Add_InfiniOps_ILUVATAR");
#endif
#ifdef USE_METAX
REGISTER_KERNEL(Device::METAX, OpType::Add, AddInfiniOps,
                "Add_InfiniOps_METAX");
#endif
#ifdef USE_MOORE
REGISTER_KERNEL(Device::MOORE, OpType::Add, AddInfiniOps,
                "Add_InfiniOps_MOORE");
#endif

} // namespace infini

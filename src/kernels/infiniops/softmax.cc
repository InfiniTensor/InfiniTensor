#include "infiniops_common.h"
#include "operators/softmax.h"

#include <base/softmax_infinilm.h>
#include <optional>

namespace infini {

namespace {

class SoftmaxInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<SoftmaxObj>(_op);
        auto input = infiniops::toInfiniOpsTensor(op->getInputs(0), context);
        auto output = infiniops::toInfiniOpsTensor(op->getOutput(), context);
        auto handle = infiniops::makeInfiniOpsHandle(context);
        ::infini::ops::SoftmaxInfinilm::Call(
            handle, {}, input, static_cast<int64_t>(op->getAxis()),
            std::optional<::infini::rt::DataType>{}, output);
    }
};

} // namespace

#ifdef USE_CUDA
REGISTER_KERNEL(Device::CUDA, OpType::Softmax, SoftmaxInfiniOps,
                "Softmax_InfiniOps_CUDA");
#endif

} // namespace infini

#include "core/kernel.h"
#include "operators/unary.h"

namespace infini {
namespace {

template <typename From, typename To>
void CastValues(const Operator &op) {
    const auto input = op->getInputs(0);
    const auto output = op->getOutput();
    const auto *source = input->getRawDataPtr<const From *>();
    auto *target = output->getRawDataPtr<To *>();
    for (size_t i = 0; i < output->size(); ++i)
        target[i] = static_cast<To>(source[i]);
}

class CastCpu : public CpuKernelWithoutConfig {
    void compute(const Operator &op, const RuntimeObj *) const override {
        const auto cast = as<CastObj>(op);
        switch (cast->getType()) {
        case CastType::Int322Int64: CastValues<int32_t, int64_t>(op); break;
        case CastType::Int642Int32: CastValues<int64_t, int32_t>(op); break;
        case CastType::Int642Float: CastValues<int64_t, float>(op); break;
        case CastType::Float2Int64: CastValues<float, int64_t>(op); break;
        case CastType::Float2Int32: CastValues<float, int32_t>(op); break;
        case CastType::Float2Float: CastValues<float, float>(op); break;
        default: IT_TODO_HALT();
        }
    }
};

} // namespace
REGISTER_KERNEL(Device::CPU, OpType::Cast, CastCpu, "Cast_CPU");
} // namespace infini

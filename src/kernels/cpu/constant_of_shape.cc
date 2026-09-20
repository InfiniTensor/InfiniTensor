#include "core/kernel.h"
#include "operators/unary.h"

namespace infini {
namespace {
class ConstantOfShapeCpu : public CpuKernelWithoutConfig {
    template <typename T> static void fill(const Operator &op) {
        auto constant = as<ConstantOfShapeObj>(op);
        auto *output = constant->getOutput()->getRawDataPtr<T *>();
        std::fill(output, output + constant->getOutput()->size(),
                  static_cast<T>(constant->getValue()));
    }
    void compute(const Operator &op, const RuntimeObj *) const override {
        switch (op->getOutput()->getDType().getIndex()) {
        case 1: fill<float>(op); break;
        case 6: fill<int32_t>(op); break;
        case 7: fill<int64_t>(op); break;
        default: IT_TODO_HALT_MSG("Unsupported ConstantOfShape dtype");
        }
    }
};
} // namespace
REGISTER_KERNEL(Device::CPU, OpType::ConstantOfShape, ConstantOfShapeCpu,
                "ConstantOfShape_CPU");
} // namespace infini

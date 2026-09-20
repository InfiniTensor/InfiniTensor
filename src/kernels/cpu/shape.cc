#include "core/kernel.h"
#include "operators/unary.h"

namespace infini {

class ShapeCPU final : public CpuKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<ShapeObj>(_op);
        IT_ASSERT(op != nullptr);
        IT_ASSERT(op->getOutput()->getDType() == DataType::Int64);

        const Shape inputShape = op->getInputs(0)->getDims();
        auto *output = op->getOutput()->getRawDataPtr<int64_t *>();

        for (size_t i = 0; i < inputShape.size(); ++i) {
            output[i] = static_cast<int64_t>(inputShape[i]);
        }
    }
};
REGISTER_KERNEL(Device::CPU, OpType::Shape, ShapeCPU, "Shape_CPU");
} // namespace infini
#include "operators/reshape.h"
#include "core/kernel.h"
#include "operators/squeeze.h"
#include "operators/unsqueeze.h"
#include "operators/unary.h"

namespace infini {
class NaiveIdentity : public CpuKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto size = _op->getInputs()[0]->getBytes();
        void *inptr = _op->getInputs(0)->getRawDataPtr<void *>();
        void *outptr = _op->getOutput()->getRawDataPtr<void *>();

        std::memcpy(outptr, inptr, size);
    }
};

class ShapeCpu : public CpuKernelWithoutConfig {
    void compute(const Operator &op, const RuntimeObj *) const override {
        auto shape = as<ShapeObj>(op);
        auto *output = shape->getOutput()->getRawDataPtr<int64_t *>();
        const auto dims = shape->getInputs(0)->getDims();
        for (size_t i = 0; i < dims.size(); ++i)
            output[i] = static_cast<int64_t>(dims[i]);
    }
};

REGISTER_KERNEL(Device::CPU, OpType::Reshape, NaiveIdentity,
                "ReshapeNaive_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Identity, NaiveIdentity,
                "IdentityNaive_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Unsqueeze, NaiveIdentity,
                "UnsqueezeNaive_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Squeeze, NaiveIdentity,
                "SqueezeNaive_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Flatten, NaiveIdentity,
                "FlattenNaive_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Shape, ShapeCpu, "Shape_CPU");

} // namespace infini

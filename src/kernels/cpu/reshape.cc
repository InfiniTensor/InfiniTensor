#include "operators/reshape.h"
#include "core/kernel.h"
#include "operators/squeeze.h"
#include "operators/unsqueeze.h"

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

REGISTER_KERNEL(Device(Device::Type::kCpu), OpType::Reshape, NaiveIdentity,
                "ReshapeNaive_CPU");
REGISTER_KERNEL(Device(Device::Type::kCpu), OpType::Identity, NaiveIdentity,
                "IdentityNaive_CPU");
REGISTER_KERNEL(Device(Device::Type::kCpu), OpType::Unsqueeze, NaiveIdentity,
                "UnsqueezeNaive_CPU");
REGISTER_KERNEL(Device(Device::Type::kCpu), OpType::Squeeze, NaiveIdentity,
                "SqueezeNaive_CPU");
REGISTER_KERNEL(Device(Device::Type::kCpu), OpType::Flatten, NaiveIdentity,
                "FlattenNaive_CPU");

} // namespace infini

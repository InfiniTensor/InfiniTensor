#include "operators/GBMML.h"
#include "core/kernel.h"
#include "custom_ops.h"

namespace infini {

template <typename T> class NaiveGBMML : public Kernel {
    void compute(const Operator &_op, const PerfRecord &record,
                 const RuntimeObj *context) const override {
        auto op = as<GBMMLObj>(_op);
    }

    void compute(const Operator &op, const RuntimeObj *context) const override {
        compute(op, {}, context);
    }

    PerfRecord tune(const Operator &op,
                    const RuntimeObj *context) const override {
        return PerfRecord(timeit([&]() { compute(op, context); }));
    }
};

REGISTER_KERNEL(Device::CPU, OpType::GBMML, DataType::UInt32,
                NaiveGBMML<uint32_t>, "GBMMLNaive_CPU_uint32");

REGISTER_KERNEL(Device::CPU, OpType::GBMML, DataType::Float32,
                NaiveGBMML<float>, "GBMMLNaive_CPU_float32");

} // namespace infini
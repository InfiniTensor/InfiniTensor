#include "operators/GBMM.h"
#include "core/kernel.h"
#include "custom_ops.h"

namespace infini {

template <typename T> class NaiveGBMM : public Kernel {
    void compute(const Operator &_op, const PerfRecord &record,
                 const RuntimeObj *context) const override {
        auto op = as<GBMMObj>(_op);
    }

    void compute(const Operator &op, const RuntimeObj *context) const override {
        compute(op, {}, context);
    }

    PerfRecord tune(const Operator &op,
                    const RuntimeObj *context) const override {
        return PerfRecord(timeit([&]() { compute(op, context); }));
    }
};

REGISTER_KERNEL(Device::CPU, OpType::GBMM, DataType::UInt32,
                NaiveGBMM<uint32_t>, "GBMMNaive_CPU_uint32");

REGISTER_KERNEL(Device::CPU, OpType::GBMM, DataType::Float32,
                NaiveGBMM<float>, "GBMMNaive_CPU_float32");

} // namespace infini
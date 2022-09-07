#include "operators/G2BMM.h"
#include "core/kernel.h"
#include "custom_ops.h"

namespace infini {

template <typename T> class NaiveG2BMM : public Kernel {
    void compute(const Operator &_op, const PerfRecord &record,
                 const RuntimeObj *context) const override {
        auto op = as<G2BMMObj>(_op);
    }

    void compute(const Operator &op, const RuntimeObj *context) const override {
        compute(op, {}, context);
    }

    PerfRecord tune(const Operator &op,
                    const RuntimeObj *context) const override {
        return PerfRecord(timeit([&]() { compute(op, context); }));
    }
};

REGISTER_KERNEL(Device::CPU, OpType::G2BMM, DataType::UInt32,
                NaiveG2BMM<uint32_t>, "G2BMMNaive_CPU_uint32");

REGISTER_KERNEL(Device::CPU, OpType::G2BMM, DataType::Float32,
                NaiveG2BMM<float>, "G2BMMNaive_CPU_float32");

} // namespace infini
#include "core/kernel.h"
#include "operators/unary.h"

namespace infini {

class NativeCast : public CpuKernelWithoutConfig {
    template <typename From, typename To>
    void doCompute(const Operator &_op, const RuntimeObj *) const {
        const auto op = as<CastObj>(_op);
        const From *input = op->getInputs(0)->getRawDataPtr<From *>();
        To *output = op->getOutput()->getRawDataPtr<To *>();
        const auto n = op->getOutput()->size();
        for (size_t i = 0; i < n; ++i) {
            output[i] = static_cast<To>(input[i]);
        }
    }

    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
#define CASE(name, From, To)                                                   \
    case CastType::name:                                                       \
        doCompute<From, To>(_op, context);                                     \
        break

        switch (as<CastObj>(_op)->getType()) {
            CASE(Float2Int64, float, int64_t);
            CASE(Float2Int32, float, int32_t);
            CASE(Float2Int16, float, int16_t);
            CASE(Float2Int8, float, int8_t);
            CASE(Float2Float, float, float);
            CASE(Int322Float, int32_t, float);
            CASE(Int322Int8, int32_t, int8_t);
            CASE(Int322Int16, int32_t, int16_t);
            CASE(Int322Int64, int32_t, int64_t);
            CASE(Int162Float, int16_t, float);
            CASE(Int162Int32, int16_t, int32_t);
            CASE(Int82Float, int8_t, float);
            CASE(Int82Int16, int8_t, int16_t);
            CASE(Int82Int32, int8_t, int32_t);
            CASE(Uint82Float, uint8_t, float);
            CASE(Uint82Int32, uint8_t, int32_t);
            CASE(Uint82Int64, uint8_t, int64_t);
            CASE(Int642Int32, int64_t, int32_t);
            CASE(Int642Uint32, int64_t, uint32_t);
            CASE(Int642Float, int64_t, float);
            CASE(Uint322Int64, uint32_t, int64_t);
        default:
            // The remaining conversions are the half-precision ones. Neither
            // Float16 nor BFloat16 has an arithmetic form on the host -- both
            // are carried as a bare pair of bytes -- so converting one is bit
            // manipulation rather than a cast, and is left undone here rather
            // than done wrongly. Such a cast halted before this kernel existed
            // and still does.
            IT_TODO_HALT();
        }

#undef CASE
    }
};

REGISTER_KERNEL(Device::CPU, OpType::Cast, NativeCast, "Cast_CPU");

} // namespace infini

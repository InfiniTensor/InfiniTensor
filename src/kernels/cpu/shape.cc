#include "core/kernel.h"
#include "operators/gather.h"
#include "operators/unary.h"
#include <cmath>
#include <cstring>
#include <limits>

namespace infini {

class ShapeCpu : public CpuKernelWithoutConfig {
    void compute(const Operator &op, const RuntimeObj *) const override {
        const auto shape = op->getInputs(0)->getDims();
        op->getOutput()->copyin(vector<int64_t>(shape.begin(), shape.end()));
    }
};

class GatherCpu : public CpuKernelWithoutConfig {
    void compute(const Operator &op, const RuntimeObj *) const override {
        const auto data = op->getInputs(0), indices = op->getInputs(1);
        const auto dims = data->getDims();
        const int axis = as<GatherObj>(op)->getAxis();
        vector<int64_t> values;
        if (indices->getDType() == DataType::Int64)
            values = indices->copyout<int64_t>();
        else {
            IT_ASSERT(indices->getDType() == DataType::Int32,
                      "Gather indices must be int32 or int64");
            auto small = indices->copyout<int32_t>();
            values.assign(small.begin(), small.end());
        }
        size_t outer = 1, inner = data->getDType().getSize();
        for (int i = 0; i < axis; ++i)
            outer *= dims[i];
        for (size_t i = axis + 1; i < dims.size(); ++i)
            inner *= dims[i];
        for (auto &index : values) {
            IT_ASSERT(index >= -dims[axis] && index < dims[axis],
                      "Gather index is out of bounds");
            if (index < 0)
                index += dims[axis];
        }
        const auto src = data->getRawDataPtr<const uint8_t *>();
        const auto dst = op->getOutput()->getRawDataPtr<uint8_t *>();
        for (size_t i = 0; i < outer; ++i)
            for (size_t j = 0; j < values.size(); ++j)
                std::memcpy(dst + (i * values.size() + j) * inner,
                            src + (i * dims[axis] + values[j]) * inner, inner);
    }
};

class CastCpu : public CpuKernelWithoutConfig {
    template <class In, class Out> void cast(const Operator &op) const {
        auto src = op->getInputs(0)->getRawDataPtr<const In *>();
        auto dst = op->getOutput()->getRawDataPtr<Out *>();
        for (size_t i = 0; i < op->getOutput()->size(); ++i) {
            if constexpr (std::is_floating_point_v<In> &&
                          std::is_integral_v<Out>) {
                const long double value = src[i];
                IT_ASSERT(std::isfinite(value) &&
                              value >= std::numeric_limits<Out>::lowest() &&
                              value <= std::numeric_limits<Out>::max(),
                          "Cast float-to-integer value out of range");
            }
            dst[i] = static_cast<Out>(src[i]);
        }
    }
    template <class In> void dispatchOutput(const Operator &op) const {
        switch (op->getOutDType().getIndex()) {
        case 1:
            cast<In, float>(op);
            break;
        case 6:
            cast<In, int32_t>(op);
            break;
        case 7:
            cast<In, int64_t>(op);
            break;
        default:
            IT_ASSERT(false, "CPU Cast supports float32/int32/int64");
        }
    }
    void compute(const Operator &op, const RuntimeObj *) const override {
        switch (op->getDType().getIndex()) {
        case 1:
            dispatchOutput<float>(op);
            break;
        case 6:
            dispatchOutput<int32_t>(op);
            break;
        case 7:
            dispatchOutput<int64_t>(op);
            break;
        default:
            IT_ASSERT(false, "CPU Cast supports float32/int32/int64");
        }
    }
};

REGISTER_KERNEL(Device::CPU, OpType::Shape, ShapeCpu, "Shape_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Gather, GatherCpu, "Gather_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Cast, CastCpu, "Cast_CPU");

} // namespace infini

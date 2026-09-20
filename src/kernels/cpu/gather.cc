#include "core/kernel.h"
#include "operators/gather.h"

namespace infini {
namespace {

class GatherCpu : public CpuKernelWithoutConfig {
    template <typename T>
    static void run(const Operator &operator_ref) {
        auto op = as<GatherObj>(operator_ref);
        const auto data = op->getInputs(0);
        const auto indices = op->getInputs(1);
        const auto output = op->getOutput();
        const auto data_dims = data->getDims();
        const auto index_dims = indices->getDims();
        const auto output_dims = output->getDims();
        const int axis = op->getAxis();
        const auto data_stride = data->getStride();
        const auto index_stride = indices->getStride();
        const auto output_stride = output->getStride();
        const T *data_ptr = data->getRawDataPtr<const T *>();
        const int64_t *index64 = nullptr;
        const int32_t *index32 = nullptr;
        if (indices->getDType() == DataType::Int64)
            index64 = indices->getRawDataPtr<const int64_t *>();
        else {
            IT_ASSERT(indices->getDType() == DataType::Int32);
            index32 = indices->getRawDataPtr<const int32_t *>();
        }
        T *out_ptr = output->getRawDataPtr<T *>();
        for (size_t flat = 0; flat < output->size(); ++flat) {
            size_t remainder = flat;
            size_t input_offset = 0;
            size_t index_offset = 0;
            for (size_t d = 0; d < output_dims.size(); ++d) {
                const int coordinate = static_cast<int>(remainder / output_stride[d]);
                remainder %= output_stride[d];
                if (d < static_cast<size_t>(axis)) {
                    input_offset += static_cast<size_t>(coordinate) * data_stride[d];
                } else if (d < static_cast<size_t>(axis) + index_dims.size()) {
                    const size_t index_dim = d - static_cast<size_t>(axis);
                    index_offset += static_cast<size_t>(coordinate) * index_stride[index_dim];
                } else {
                    const size_t input_dim = d - index_dims.size() + 1;
                    input_offset += static_cast<size_t>(coordinate) * data_stride[input_dim];
                }
            }
            const int64_t index_value = index64 ? index64[index_offset] : index32[index_offset];
            IT_ASSERT(index_value >= 0 && index_value < data_dims[axis]);
            input_offset += static_cast<size_t>(index_value) * data_stride[axis];
            out_ptr[flat] = data_ptr[input_offset];
        }
    }

    void compute(const Operator &op, const RuntimeObj *) const override {
        switch (op->getDType().getIndex()) {
        case 1: run<float>(op); break;
        case 6: run<int32_t>(op); break;
        case 7: run<int64_t>(op); break;
        default: IT_TODO_HALT();
        }
    }
};

} // namespace

REGISTER_KERNEL(Device::CPU, OpType::Gather, GatherCpu, "Gather_CPU");
} // namespace infini

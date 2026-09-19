#include "operators/gather.h"
#include "core/kernel.h"

namespace infini {

/// Picks entries out of the input along one axis, as ONNX Gather describes.
///
/// The input is read as three nested runs -- whatever lies before the axis,
/// the axis itself, and whatever lies after it -- because that is the only
/// part of the layout this has to care about: an entry is picked by its
/// position along the axis, and the run of elements behind that position comes
/// along with it untouched.
class NaiveGather : public CpuKernelWithoutConfig {
    /// Reads one index, which ONNX allows to be either width of integer.
    static int64_t indexAt(const Tensor &indices, size_t at) {
        if (indices->getDType() == DataType::Int64) {
            return indices->getRawDataPtr<int64_t *>()[at];
        }
        return indices->getRawDataPtr<int32_t *>()[at];
    }

    template <typename T>
    void doCompute(const Operator &_op, const RuntimeObj *context) const {
        auto op = as<GatherObj>(_op);
        const auto &dims = op->getInputs(0)->getDims();
        const auto axis = (size_t)op->getAxis();

        size_t outer = 1;
        for (size_t i = 0; i < axis; ++i) {
            outer *= dims[i];
        }
        size_t inner = 1;
        for (size_t i = axis + 1; i < dims.size(); ++i) {
            inner *= dims[i];
        }
        const size_t along = dims[axis];

        const auto &indices = op->getInputs(1);
        const size_t picked = indices->size();
        auto inPtr = op->getInputs(0)->getRawDataPtr<T *>();
        auto outPtr = op->getOutput()->getRawDataPtr<T *>();

        for (size_t k = 0; k < picked; ++k) {
            // ONNX counts a negative index back from the end of the axis.
            auto index = indexAt(indices, k);
            const size_t at = index < 0 ? index + (int64_t)along : index;
            IT_ASSERT(at < along, "gather index out of range");
#pragma omp parallel for
            for (size_t o = 0; o < outer; ++o) {
                std::memcpy(outPtr + (o * picked + k) * inner,
                            inPtr + (o * along + at) * inner,
                            inner * sizeof(T));
            }
        }
    }

    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
#define CASE(N)                                                                \
    case N:                                                                    \
        doCompute<DT<N>::t>(_op, context)

        int dataTypeIdx = _op->getDType().getIndex();
        switch (dataTypeIdx) {
            CASE(1); // DataType::Float32
            break;
            CASE(6); // DataType::Int32
            break;
            CASE(7); // DataType::Int64
            break;
            CASE(11); // DataType::Double
            break;
            CASE(12); // DataType::UInt32
            break;
        default:
            IT_TODO_HALT();
        }
    }
};

REGISTER_KERNEL(Device::CPU, OpType::Gather, NaiveGather, "GatherNaive_CPU");

} // namespace infini

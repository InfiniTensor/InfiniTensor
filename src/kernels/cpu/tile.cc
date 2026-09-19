#include "operators/tile.h"
#include "core/kernel.h"

namespace infini {

class NaiveTile : public CpuKernelWithoutConfig {
    template <typename T>
    void doCompute(const Operator &_op, const RuntimeObj *context) const {
        auto op = as<TileObj>(_op);
        const auto &input = op->getInputs(0);
        const auto &output = op->getOutput();
        const T *inPtr = input->getRawDataPtr<T *>();
        T *outPtr = output->getRawDataPtr<T *>();

        const Shape &inDims = input->getDims();
        const Shape &outDims = output->getDims();
        IT_ASSERT(outDims.size() == inDims.size());

        const size_t rank = inDims.size();
        Shape inStride(rank, 1), outStride(rank, 1);
        for (size_t i = rank; i > 1; --i) {
            inStride[i - 2] = inStride[i - 1] * inDims[i - 1];
            outStride[i - 2] = outStride[i - 1] * outDims[i - 1];
        }

        const size_t total = output->size();
        for (size_t o = 0; o < total; ++o) {
            // Which element of the input this one repeats: the position along
            // each axis wraps back to the start of the input every time it runs
            // past the end, which is what repeating that axis means.
            size_t in = 0, rest = o;
            for (size_t axis = 0; axis < rank; ++axis) {
                const size_t at = rest / static_cast<size_t>(outStride[axis]);
                rest %= static_cast<size_t>(outStride[axis]);
                in += (at % static_cast<size_t>(inDims[axis])) *
                      static_cast<size_t>(inStride[axis]);
            }
            outPtr[o] = inPtr[in];
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

REGISTER_KERNEL(Device::CPU, OpType::Tile, NaiveTile, "TileNaive_CPU");

} // namespace infini

#include "operators/split.h"
#include "core/kernel.h"

namespace infini {

class NaiveSplit : public CpuKernelWithoutConfig {
    template <typename T>
    void doCompute(const Operator &_op, const RuntimeObj *context) const {
        auto op = as<SplitObj>(_op);
        auto inputs = op->getInputs(), outputs = op->getOutputs();
        auto dim = op->getDim();
        auto input = inputs[0];
        const auto &inDim = input->getDims();
        std::vector<Shape> outDims;
        for (auto output : outputs)
            outDims.emplace_back(output->getDims());
        size_t blockOffsetInner = 1;
        for (size_t i = inDim.size() - 1; i > (size_t)dim; --i)
            blockOffsetInner *= inDim[i];
        size_t blockOffset = inDim[dim] * blockOffsetInner;
        for (size_t i = 0; i < outputs.size(); ++i) {
            auto output = outputs[i];
            auto dimOffset = 0;
            auto outDim = outDims[i];
            for (size_t j = 0; j < i; ++j)
                dimOffset += outDims[j][dim];
            size_t localBlockOffset = 1;
            for (size_t i = outDim.size() - 1;
                 i >= (size_t)dim && i != (size_t)-1; --i)
                localBlockOffset *= outDim[i];
            auto innerOffset = blockOffsetInner * dimOffset;
            auto outSize = output->size();
            auto inPtr = input->getRawDataPtr<T *>(),
                 outPtr = output->getRawDataPtr<T *>();
#pragma omp parallel for
            for (size_t oOffset = 0; oOffset < outSize; ++oOffset) {
                auto iOffset = oOffset % localBlockOffset + innerOffset +
                               oOffset / localBlockOffset * blockOffset;
                outPtr[oOffset] = inPtr[iOffset];
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
            CASE(12); // DataType::UInt32
            break;
        default:
            IT_TODO_HALT();
        }
    }
};

REGISTER_KERNEL(Device::CPU, OpType::Split, NaiveSplit, "SplitNaive_CPU");

} // namespace infini

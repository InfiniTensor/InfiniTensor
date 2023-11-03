#include "operators/split.h"
#include "core/kernel.h"

namespace infini {

template <typename T> class NaiveSplit : public CpuKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
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
            auto inPtr = input->getRawDataPtr<T *>(),
                 outPtr = output->getRawDataPtr<T *>();

            // MSVC: index variable in OpenMP 'for' statement must have signed
            // integral type
            long long outSize = static_cast<long long>(output->size());
#pragma omp parallel for
            for (long long oOffset = 0; oOffset < outSize; ++oOffset) {
                auto iOffset = oOffset % localBlockOffset + innerOffset +
                               oOffset / localBlockOffset * blockOffset;
                outPtr[oOffset] = inPtr[iOffset];
            }
        }
    }
};

REGISTER_KERNEL(Device::CPU, OpType::Split, DataType::UInt32,
                NaiveSplit<uint32_t>, "SplitNaive_CPU_uint32");
REGISTER_KERNEL(Device::CPU, OpType::Split, DataType::Float32,
                NaiveSplit<float>, "SplitNaive_CPU_float32");

} // namespace infini

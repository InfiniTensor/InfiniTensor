#include "operators/concat.h"
#include "core/kernel.h"

namespace infini {

template <typename T> class NaiveConcat : public CpuKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<ConcatObj>(_op);
        auto inputs = op->getInputs(), outputs = op->getOutputs();
        auto dim = op->getDim();
        auto output = outputs[0];
        std::vector<Shape> iDims;
        for (auto input : inputs)
            iDims.emplace_back(input->getDims());
        const auto &outDim = output->getDims();
        size_t blockOffsetInner = 1;
        for (size_t i = outDim.size() - 1; i > (size_t)dim; --i)
            blockOffsetInner *= outDim[i];
        size_t blockOffset = outDim[dim] * blockOffsetInner;
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto input = inputs[i];
            auto dimOffset = 0;
            auto iDim = iDims[i];
            for (size_t j = 0; j < i; ++j)
                dimOffset += iDims[j][dim];
            size_t localBlockOffset = 1;
            for (size_t i = iDim.size() - 1;
                 i >= (size_t)dim && i != (size_t)-1; --i)
                localBlockOffset *= iDim[i];
            auto innerOffset = blockOffsetInner * dimOffset;
            auto inPtr = input->getRawDataPtr<T *>(),
                 outPtr = output->getRawDataPtr<T *>();

            // MSVC: index variable in OpenMP 'for' statement must have signed
            // integral type
            long long inSize = static_cast<long long>(input->size());
#pragma omp parallel for
            for (long long iOffset = 0; iOffset < inSize; ++iOffset) {
                auto oOffset = iOffset % localBlockOffset + innerOffset +
                               iOffset / localBlockOffset * blockOffset;
                // output->setData(oOffset, input->getData(iOffset));
                outPtr[oOffset] = inPtr[iOffset];
            }
        }
    }
};

REGISTER_KERNEL(Device::CPU, OpType::Concat, DataType::UInt32,
                NaiveConcat<uint32_t>, "ConcatNaive_CPU_uint32");
REGISTER_KERNEL(Device::CPU, OpType::Concat, DataType::Float32,
                NaiveConcat<float>, "ConcatNaive_CPU_float32");

} // namespace infini

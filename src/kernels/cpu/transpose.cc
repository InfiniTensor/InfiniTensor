#include "operators/transpose.h"
#include "core/kernel.h"

namespace infini {

inline Shape idx2Pos(const Shape &shape, size_t idx) {
    Shape pos = Shape(shape.size(), 0);
    auto rest = idx, curDimId = shape.size() - 1;
    while (rest > 0) {
        pos[curDimId] = rest % shape[curDimId];
        rest /= shape[curDimId];
        curDimId--;
    }
    return pos;
}

template <typename T> class NaiveTranspose : public CpuKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<TransposeObj>(_op);
        auto inputs = op->getInputs(), outputs = op->getOutputs();
        const auto &inDim = inputs[0]->getDims();
        const auto &perm = op->getPermute();

        size_t inSize = inputs[0]->size();
        auto inPtr = inputs[0]->getRawDataPtr<T *>(),
             outPtr = outputs[0]->getRawDataPtr<T *>();
        // #pragma omp parallel for
        for (size_t inIdx = 0; inIdx < inSize; ++inIdx) {
            auto posInput = idx2Pos(inDim, inIdx);
            int outIdx = 0;
            for (size_t j = 0, jEnd = perm.size(); j < jEnd; ++j) {
                outIdx = outIdx * inDim[perm[j]] + posInput[perm[j]];
            }
            outPtr[outIdx] = inPtr[inIdx];
        }
    }
};

REGISTER_KERNEL(Device::CPU, OpType::Transpose, DataType::UInt32,
                NaiveTranspose<uint32_t>, "TransposeNaive_CPU_uint32");
REGISTER_KERNEL(Device::CPU, OpType::Transpose, DataType::Float32,
                NaiveTranspose<float>, "TransposeNaive_CPU_float32");

} // namespace infini

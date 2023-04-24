#include "operators/transpose.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_transpose.h"

namespace infini {

class TransposeCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<TransposeObj>(_op);

        auto input = op->getInputs(0);
        auto output = op->getOutput();
        void *const inputData = input->getRawDataPtr<void *>();
        void *const outputData = output->getRawDataPtr<void *>();
        const auto &inputShape = input->getDims();
        const auto &outputShape = output->getDims();

        const auto &perm = op->getPermute();
        int size = input->size();
        int nDims = input->getDims().size();

        // Compute strides
        SmallArray strides, buffer;
        IT_ASSERT(nDims <= SMALL_ARRAY_SIZE);
        int curStride = 1;
        for (int i = nDims - 1; i >= 0; --i) {
            buffer.data[i] = curStride;
            curStride *= inputShape[i];
        }
        for (int i = 0; i < nDims; ++i) {
            strides.data[i] = buffer.data[perm[i]];
        }

        SmallArray outputDims;
        for (int i = 0; i < nDims; ++i) {
            outputDims.data[i] = outputShape[i];
        }

        transpose_kernel((float *)inputData, (float *)outputData, nDims, size,
                         strides, outputDims, input->getDims(),
                         output->getDims(), perm);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Transpose, DataType::Float32,
                TransposeCuda, "Transpose_CUDA_Float32");

} // namespace infini

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

        const int dType = op->getDType().getIndex();
        bool condition = false;
        if (dType == 1 && nDims == 4) {
            if (perm[0] == 0 && perm[1] == 2 && perm[2] == 1 && perm[3] == 3) {
                condition = true;
            }
        }

        // std::cout << "transpose: " << dType << std::endl;
        // for (int i = 0; i < nDims; i++) {
        //     printf("%d ", inputShape[i]);
        // }
        // printf("\n");
        // for (int i = 0; i < nDims; i++) {
        //     printf("%d ", outputShape[i]);
        // }
        // printf("\n");
        if (condition) {
            transpose_nchw2nhcw(inputData, outputData, inputShape[0],
                                inputShape[1], inputShape[2], inputShape[3]);
        } else {
            transpose_kernel(dType, inputData, outputData, nDims, size, strides,
                             outputDims);
        }
    }
};

class DepthToSpaceCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<DepthToSpaceObj>(_op);

        auto input = op->getInputs(0);
        auto output = op->getOutput();
        void *const inputData = input->getRawDataPtr<void *>();
        void *const outputData = output->getRawDataPtr<void *>();
        const auto &reshape = op->getReshapeDim();
        const auto &transpose = op->getTransposeDim();
        auto mode = op->getMode();

        std::vector<int> perm;
        if (mode == 0) {
            perm = {0, 3, 4, 1, 5, 2};
        } else {
            perm = {0, 1, 4, 2, 5, 3};
        }

        int size = input->size();
        int nDims = reshape.size();

        // Compute strides
        SmallArray strides, buffer;
        IT_ASSERT(nDims <= SMALL_ARRAY_SIZE);
        int curStride = 1;
        for (int i = nDims - 1; i >= 0; --i) {
            buffer.data[i] = curStride;
            curStride *= reshape[i];
        }
        for (int i = 0; i < nDims; ++i) {
            strides.data[i] = buffer.data[perm[i]];
        }

        SmallArray outputDims;
        for (int i = 0; i < nDims; ++i) {
            outputDims.data[i] = transpose[i];
        }
        const int dType = op->getDType().getIndex();
        transpose_kernel(dType, inputData, outputData, nDims, size, strides,
                         outputDims);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Transpose, TransposeCuda,
                "Transpose_CUDA");

REGISTER_KERNEL(Device::CUDA, OpType::DepthToSpace, DepthToSpaceCuda,
                "DepthToSpace_CUDA");

} // namespace infini

#include "operators/softmax.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_softmax.h"

namespace infini {
class SoftmaxCuda : public CudaKernelWithoutConfig {

    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<SoftmaxObj>(_op);
        auto input = op->getInputs(0)->getRawDataPtr<float *>();
        auto output = op->getOutput(0)->getRawDataPtr<float *>();
        const auto &inShape = op->getInputs(0)->getDims(); // input shape
        int nDims = op->getInputs(0)->getDims().size();
        IT_ASSERT(nDims <= SMALL_ARRAY_SIZE);
        int size = 1;             // size = i(JKS) + j(KS) + k(S) + s
        int stride = 1, temp = 1; // stride=[JKS, KS, S, 1][axis]
        int axis = op->getAxis();
        SmallArray inputShape;
        for (int i = nDims - 1; i >= 0;
             --i) { // must i = nDims - 1, --i; can't i = 0, i++
            size *= inShape[i];
            inputShape.data[i] = inShape[i];
            if (i == axis) {
                stride = temp;
            }
            temp *= inShape[i];
        }

        softmax_kernel((float *)input, (float *)output, size, inputShape, axis,
                       nDims, stride);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Softmax, DataType::Float32, SoftmaxCuda,
                "Softmax_CUDA_Float32");
} // namespace infini

#include "operators/softmax.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_softmax.h"

namespace infini {
class SoftmaxCudnn : public CudaKernelWithoutConfig {

    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<SoftmaxObj>(_op);
        auto input = op->getInputs(0)->getRawDataPtr<float *>();
        auto output = op->getOutput(0)->getRawDataPtr<float *>();
        const auto &in_Shape = op->getInputs(0)->getDims(); // input shape
        int nDims = op->getInputs(0)->getDims().size();
        IT_ASSERT(nDims <= SMALL_ARRAY_SIZE);
        int size = 1;
        SmallArray inputShape;
        for (int i = 0; i < nDims; ++i) {
            size *= in_Shape[i];
            inputShape.data[i] = in_Shape[i];
        }

        int axis = op->getAxis();

        softmax_kernel((float *)input, (float *)output, size, inputShape, axis,
                       nDims);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Softmax, DataType::Float32, SoftmaxCudnn,
                "Softmax_CUDA_Float32");
} // namespace infini

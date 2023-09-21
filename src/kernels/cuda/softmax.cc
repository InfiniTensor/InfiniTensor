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
        const auto &inputShape = op->getInputs(0)->getDims(); // input shape
        int nDims = op->getInputs(0)->getDims().size();
        IT_ASSERT(nDims <= SMALL_ARRAY_SIZE);
        int size = 1;

        for (int i = 0; i < nDims; ++i) {
            size *= inputShape[i];
        }
        auto dims = op->getInputs(0)->getDims();
        int dimsize = dims[op->getAxis()];
        int size_y = size / dimsize;
        int stride = op->getInputs(0)->getStride().at(op->getAxis());
        softmax_kernel((float *)input, (float *)output, size, size_y, dimsize,
                       stride);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Softmax, DataType::Float32, SoftmaxCuda,
                "Softmax_CUDA_Float32");
} // namespace infini

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
        auto dims = op->getInputs(0)->getDims();
        int nDims = dims.size();
        int size = 1; // size = i(JKS) + j(KS) + k(S) + s
        int dimsize = dims[op->getAxis()];
        int stride = op->getInputs(0)->getStride().at(op->getAxis());
        for (int i = nDims - 1; i >= 0;
             --i) { // must i = nDims - 1, --i; can't i = 0, i++
            size *= inShape[i];
        }
        int num_blocks = size / dimsize;
        softmax_kernel(num_blocks, (float *)input, (float *)output, size,
                       dimsize, stride);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Softmax, DataType::Float32, SoftmaxCuda,
                "Softmax_CUDA_Float32");
} // namespace infini

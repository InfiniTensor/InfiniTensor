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

        int size; // size = i(JKS) + j(KS) + k(S) + s
        size = op->getOutput(0)->size();
        int dimsize = dims[op->getAxis()];
        int stride = op->getInputs(0)->getStride().at(op->getAxis());

        int num_blocks = size / dimsize;
        if (op->getDType() == DataType::Float32) {
            softmax_kernel(num_blocks, (float *)input, (float *)output, size,
                           dimsize, stride);
        } else if (op->getDType() == DataType::Float16) {
            softmax_kernel(num_blocks, (half *)input, (half *)output, size,
                           dimsize, stride);
        }
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Softmax, DataType::Float32, SoftmaxCuda,
                "Softmax_CUDA_Float32");
} // namespace infini

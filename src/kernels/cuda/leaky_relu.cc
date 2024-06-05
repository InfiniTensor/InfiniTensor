#include "cuda/cuda_leaky_relu.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"
#include "operators/unary.h"

namespace infini {

class LeakyReluCuda : public CudaKernelWithoutConfig {

    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<LeakyReluObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        void *const inputData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const outputData = (op->getOutput()->getRawDataPtr<void *>());
        auto alphaValue = op->getAlpha();
        auto dim = op->getInputs(0)->getDims();
        int size = dim[0] * dim[1] * dim[2] * dim[3];
        leaky_relu_kernel((float *)inputData, (float *)outputData, 
                    alphaValue, size);
    }

};

REGISTER_KERNEL(Device::CUDA, OpType::LeakyRelu, LeakyReluCuda, "LeakyRelu_CUDA");

}; // namespace infini
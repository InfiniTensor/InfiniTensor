#include "operators/bias2prelu.h"
#include "cuda/cuda_bias2prelu.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"

namespace infini {

class Bias2PReluCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op, const RuntimeObj *_context) const {
        auto op = as<BiasPReLU>(_op);
        float *const input = (op->getInputs(0)->getRawDataPtr<float *>());
        float *const bias = (op->getInputs(1)->getRawDataPtr<float *>());
        float *const output = (op->getOutput()->getRawDataPtr<float *>());
        auto dim = op->getInputs(0)->getDims();
        int n = dim[0], h = dim[1], w = dim[2], c = dim[3];

        bias2prelu_kernel(input, output, bias, n, h, w, c, op->getPReLU(),
                          op->getParamReLU());
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::BiasPReLU, DataType::Float32,
                Bias2PReluCuda, "Bias2PReLU_CUDA_Float32");
} // namespace infini
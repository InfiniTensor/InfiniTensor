#include "operators/transpose.h"
#include "cuda/cuda_transpose.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"

namespace infini {

class TransposeCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<TransposeObj>(_op);
        float *const aData = (op->getInputs(0)->getRawDataPtr<float *>());
        float *const cData = (op->getOutput()->getRawDataPtr<float *>());

        auto dim = op->getInputs(0)->getDims();
        int n = dim[0], c = dim[1], h = dim[2], w = dim[3];
        auto permute = op->getPermute();
        transpose_kernel(aData, cData, n,c,h,w, permute[0],permute[1],permute[2],permute[3]);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Transpose, DataType::Float32, TransposeCuda,
                "Transpose_CUDA_Float32");
}; // namespace infini

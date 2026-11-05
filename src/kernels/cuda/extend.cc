#include "operators/extend.h"
#include "cuda/cuda_kernel_without_config.h"

namespace infini {
void extend_kernel(float *in, float *out, int blockSize, int blockSizeOuter,
                   int oSize);
void extend_kernel(half *in, half *out, int blockSize, int blockSizeOuter,
                   int oSize);
class ExtendCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ExtendObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32 ||
                  op->getDType() == DataType::Float16);
        int blockSize = 1;
        auto iDim = op->getInputs(0)->getDims();
        for (size_t i = iDim.size() - 1;
             i >= (size_t)op->getDim() && i != (size_t)-1; --i)
            blockSize *= iDim[i];
        auto blockSizeOuter = (op->getNum() + 1) * blockSize;

        if (op->getDType() == DataType::Float16)
            extend_kernel(op->getInputs(0)->getRawDataPtr<half *>(),
                          op->getOutput()->getRawDataPtr<half *>(), blockSize,
                          blockSizeOuter, op->getOutput()->size());
        else
            extend_kernel(op->getInputs(0)->getRawDataPtr<float *>(),
                          op->getOutput()->getRawDataPtr<float *>(), blockSize,
                          blockSizeOuter, op->getOutput()->size());
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Extend, ExtendCuda, "Extend_CUDA");
} // namespace infini

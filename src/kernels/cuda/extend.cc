#include "operators/extend.h"
#include "cuda/cuda_kernel_wihtout_config.h"

namespace infini {
void extend_kernel(float *in, float *out, int blockSize, int blockSizeOuter,
                   int oSize);
class ExtendCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ExtendObj>(_op);
        auto inData = op->getInputs(0)->getRawDataPtr<float *>();
        auto outData = op->getOutputs()[0]->getRawDataPtr<float *>();
        int blockSize = 1;
        auto iDim = op->getInputs(0)->getDims();
        for (size_t i = iDim.size() - 1;
             i >= (size_t)op->getDim() && i != (size_t)-1; --i)
            blockSize *= iDim[i];
        auto blockSizeOuter = (op->getNum() + 1) * blockSize;

        extend_kernel(inData, outData, blockSize, blockSizeOuter,
                      op->getOutput()->size());
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Extend, DataType::Float32, ExtendCuda,
                "Extend_CUDA_Float32");
} // namespace infini

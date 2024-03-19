#include "cuda/cuda_kernel_wihtout_config.h"

namespace infini {
class CopyCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &op,
                 const RuntimeObj *_context) const override {
        auto inData = op->getInputs(0)->getRawDataPtr<void *>();
        auto outData = op->getOutputs()[0]->getRawDataPtr<void *>();
        cudaMemcpyAsync(outData, inData, op->getInputs(0)->getBytes(),
                        cudaMemcpyDeviceToDevice,
                        CUDAStream::getCurrentStream());
    }
};
// reshape/flatten/identity all act as copying from input to output.

REGISTER_KERNEL(Device::CUDA, OpType::Reshape, CopyCuda, "Reshape_CUDA");

REGISTER_KERNEL(Device::CUDA, OpType::Flatten, CopyCuda, "Flatten_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Identity, CopyCuda, "Identity_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Squeeze, CopyCuda, "Squeeze_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Unsqueeze, CopyCuda, "Unsqueeze_CUDA");

} // namespace infini

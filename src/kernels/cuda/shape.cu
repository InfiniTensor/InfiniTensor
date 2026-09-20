#include "core/kernel.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "operators/unary.h"

namespace infini {
namespace {

__global__ void ShapeKernel(int64_t *output, const int rank,
                            int64_t d0, int64_t d1, int64_t d2, int64_t d3,
                            int64_t d4, int64_t d5, int64_t d6, int64_t d7) {
    const int64_t dims[8] = {d0, d1, d2, d3, d4, d5, d6, d7};
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < rank)
        output[index] = dims[index];
}

class ShapeCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &op, const RuntimeObj *) const override {
        const auto input = op->getInputs(0);
        const auto output = op->getOutput();
        const auto dims = input->getDims();
        IT_ASSERT(dims.size() <= 8, "Shape supports rank up to 8");
        int64_t host_dims[8] = {};
        for (size_t i = 0; i < dims.size(); ++i) host_dims[i] = dims[i];
        ShapeKernel<<<1, 32, 0, CUDAStream::getCurrentStream()>>>(
            output->getRawDataPtr<int64_t *>(), static_cast<int>(dims.size()),
            host_dims[0], host_dims[1], host_dims[2], host_dims[3],
            host_dims[4], host_dims[5], host_dims[6], host_dims[7]);
    }
};

} // namespace
REGISTER_KERNEL(Device::CUDA, OpType::Shape, ShapeCuda, "Shape_CUDA");
} // namespace infini

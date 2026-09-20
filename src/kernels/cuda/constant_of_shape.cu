#include "core/kernel.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "operators/unary.h"

namespace infini {
namespace {
template <typename T>
__global__ void ConstantOfShapeKernel(T *output, size_t size, T value) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
                         static_cast<size_t>(threadIdx.x);
    if (index < size) output[index] = value;
}

class ConstantOfShapeCuda : public CudaKernelWithoutConfig {
    template <typename T> static void fill(const Operator &op, T value) {
        auto constant = as<ConstantOfShapeObj>(op);
        const auto size = constant->getOutput()->size();
        const int blocks = static_cast<int>((size + 255) / 256);
        ConstantOfShapeKernel<<<blocks, 256, 0, CUDAStream::getCurrentStream()>>>(
            constant->getOutput()->getRawDataPtr<T *>(), size, value);
    }
    void compute(const Operator &op, const RuntimeObj *) const override {
        const auto value = as<ConstantOfShapeObj>(op)->getValue();
        switch (op->getOutput()->getDType().getIndex()) {
        case 1: fill<float>(op, value); break;
        case 6: fill<int32_t>(op, static_cast<int32_t>(value)); break;
        case 7: fill<int64_t>(op, static_cast<int64_t>(value)); break;
        default: IT_TODO_HALT_MSG("Unsupported ConstantOfShape dtype");
        }
    }
};
} // namespace
REGISTER_KERNEL(Device::CUDA, OpType::ConstantOfShape, ConstantOfShapeCuda,
                "ConstantOfShape_CUDA");
} // namespace infini

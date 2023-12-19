#include "operators/dynamic_quantize_linear.h"
#include "cuda/cuda_dynamic_quantize_linear.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"

namespace infini {

class DynamicQuantizeLinearCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<DynamicQuantizeLinearObj>(_op);

        void *const input = (op->getInputs(0)->getRawDataPtr<void *>());

        void *const outputY = (op->getOutput(0)->getRawDataPtr<void *>());
        void *const outputYScale = (op->getOutput(1)->getRawDataPtr<void *>());
        void *const outputYZeroPoint =
            (op->getOutput(2)->getRawDataPtr<void *>());

        int size = op->getInputs(0)->size();

        dynamicQuantizeLinearKernel((float *)input, (uint8_t *)outputY,
                                    (float *)outputYScale,
                                    (uint8_t *)outputYZeroPoint, size);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::DynamicQuantizeLinear,
                DynamicQuantizeLinearCuda, "DynamicQuantizeLinear_CUDA");

}; // namespace infini
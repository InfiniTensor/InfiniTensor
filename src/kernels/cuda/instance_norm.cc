#include "operators/instance_norm.h"
#include "cuda/cuda_instancenorm.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"

namespace infini {
class InstanceNormalizationCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<InstanceNormObj>(_op);

        void *const inputData = op->getInputs(0)->getRawDataPtr<void *>();
        void *const scaleData = op->getInputs(1)->getRawDataPtr<void *>();
        void *const biasData = op->getInputs(2)->getRawDataPtr<void *>();
        void *const outputData = op->getOutput()->getRawDataPtr<void *>();

        float eps = op->getEps();

        auto dims = op->getInputs(0)->getDims();
        // 假设输入是 NCHW
        IT_ASSERT(dims.size() >= 3);

        int N = dims[0];
        int C = dims[1];
        int inner_size = 1;
        for (size_t i = 2; i < dims.size(); ++i)
            inner_size *= dims[i];

        if (op->getDType() == DataType::Float32) {
            InstanceNormKernel((const float *)inputData,
                               (const float *)scaleData,
                               (const float *)biasData, (float *)outputData, N,
                               C, inner_size, eps);
        } else if (op->getDType() == DataType::Float16) {
            InstanceNormKernel((const half *)inputData, (const half *)scaleData,
                               (const half *)biasData, (half *)outputData, N, C,
                               inner_size, eps);
        } else {
            IT_ASSERT(false);
        }
    }
};
REGISTER_KERNEL(Device::CUDA, OpType::InstanceNormalization,
                InstanceNormalizationCuda, "InstanceNorm_CUDA");

} // namespace infini
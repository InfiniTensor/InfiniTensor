#include "operators/elu.h"
#include "cuda/cuda_elu.h" // 确保包含声明
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"

namespace infini {

class EluCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<EluObj>(_op);

        void *const inputData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const outputData = (op->getOutput()->getRawDataPtr<void *>());
        auto dims = op->getInputs(0)->getDims();

        int size = 1;
        for (size_t i = 0; i < dims.size(); ++i)
            size *= dims[i];
        elu_kernel((float *)inputData, (float *)outputData, size, op->alpha);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Elu, EluCuda, "Elu_CUDA_Float32");

} // namespace infini

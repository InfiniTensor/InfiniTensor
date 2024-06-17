#include "operators/elu.h"
#include "cuda/cuda_elu.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"

namespace infini {

class EluCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<EluObj>(_op);

        void *const inputData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const outputData = (op->getOutput()->getRawDataPtr<void *>());

        int size = op->getInputs(0)->size();
        elu_kernel((float *)inputData, (float *)outputData, size, op->alpha);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Elu, EluCuda, "Elu_CUDA_Float32");

} // namespace infini

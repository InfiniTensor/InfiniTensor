#include "operators/rms_norm.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_rmsnorm.h"
#include "cuda/cuda_runtime.h"

namespace infini {

class RMSNormCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<RMSNormObj>(_op);

        auto input = op->getInputs(0);
        auto weight = op->getInputs(1);
        auto output = op->getOutput();
        void *const inputData = input->getRawDataPtr<void *>();
        void *const weightData = weight->getRawDataPtr<void *>();
        void *const outputData = output->getRawDataPtr<void *>();
        const auto &inputShape = input->getDims();
        int nDims = input->getDims().size();

        int hidden_size = inputShape[nDims - 1];
        int num_tokens = input->size() / hidden_size;
        IT_ASSERT(hidden_size == (int)weight->size());

        const int dType = op->getDType().getIndex();
        rmsnorm_kernel(dType, inputData, weightData, outputData, num_tokens,
                       hidden_size);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::RMSNorm, RMSNormCuda, "RMSNorm_CUDA");

} // namespace infini

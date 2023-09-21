#include "operators/attention.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_attention.h"


namespace infini {

class AttentionCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<AttentionObj>(_op);

        void *const inputQData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const inputKData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const inputVData = (op->getInputs(2)->getRawDataPtr<void *>());
        void *const outputData = (op->getOutput()->getRawDataPtr<void *>());
        int N = op->getInputs(0)->getDims()[0];
        int d = op->getInputs(0)->getDims()[1];

        attentionKernel((float *)inputQData, (float *)inputKData,
                    (float *)inputVData, N, d, (float *)outputData);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Attention, DataType::Float32, AttentionCuda,
                "Attention_CUDA_Float32");

}; // namespace infini

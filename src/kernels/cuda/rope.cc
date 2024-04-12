#include "operators/rope.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_rope.h"
#include "cuda/cuda_runtime.h"

namespace infini {

class RoPECuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<RoPEObj>(_op);

        auto pos = op->getInputs(0);
        auto input = op->getInputs(1);
        auto output = op->getOutput();
        void *const inputData = input->getRawDataPtr<void *>();
        void *const outputData = output->getRawDataPtr<void *>();
        const auto &inputShape = input->getDims();
        int nDims = input->getDims().size();

        IT_ASSERT(nDims == 3 && pos->getDims().size() == 2);
        IT_ASSERT(inputShape[0] == pos->getDims()[0] &&
                  inputShape[1] == pos->getDims()[1]);
        int position_idx_dtype = op->getInputs()[0]->getDTypeIndex();
        int dim_model = inputShape[2];
        int dim_head = 128; // TODO: get dim_head from the framework
        int pos_stride = inputShape[1];
        int batchsize = inputShape[0];

        const int dType = op->getDType().getIndex();
        rope_kernel(dType, pos->getRawDataPtr<int64_t *>(), inputData,
                    outputData, dim_model, dim_head, batchsize, pos_stride);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::RoPE, RoPECuda, "RoPE_CUDA");

} // namespace infini

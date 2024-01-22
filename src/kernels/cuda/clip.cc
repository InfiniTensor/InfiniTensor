#include "cuda/cuda_clip.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"
#include "operators/unary.h"

namespace infini {

class ClipCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ClipObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        void *const inputData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const outputData = (op->getOutput()->getRawDataPtr<void *>());
        auto min = op->getMin();
        auto max = op->getMax();
        auto dim = op->getInputs(0)->getDims();
        int num = dim[0] * dim[1] * dim[2] * dim[3];
        clip_kernel((float *)inputData, (float *)outputData, num,
                    min ? *min : NAN, max ? *max : NAN);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Clip, ClipCuda, "Clip_CUDA");

}; // namespace infini

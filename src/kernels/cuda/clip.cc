#include "cuda/cuda_clip.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"
#include "operators/unary.h"

namespace infini {

class ClipCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ClipObj>(_op);
        void *const inputData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const outputData = (op->getOutput()->getRawDataPtr<void *>());
        void *const min = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const max = (op->getInputs(2)->getRawDataPtr<void *>());
        auto dim = op->getInputs(0)->getDims();
        int num =
            std::accumulate(dim.begin(), dim.end(), 1, std::multiplies<int>());
        if (op->getDType() == DataType::Float32) {
            clip_kernel<float>((float *)inputData, (float *)outputData, num,
                               (float *)min, (float *)max);
        } else if (op->getDType() == DataType::Float16) {
            clip_kernel<half>((half *)inputData, (half *)outputData, num,
                              (half *)min, (half *)max);
        } else {
            IT_TODO_HALT();
        }
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Clip, ClipCuda, "Clip_CUDA");

}; // namespace infini

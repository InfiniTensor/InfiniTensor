#include "operators/unary.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_clip.h"

namespace infini {

class ClipCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ClipObj>(_op);

        void *const inputData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const outputData = (op->getOutput()->getRawDataPtr<void *>());
        float min = op->getMin();
        float max = op->getMax();
        auto dim = op->getInputs(0)->getDims();
        int num = dim[0] * dim[1] * dim[2] * dim[3];
        clip_kernel((float*)inputData, (float*)outputData, num, min, max);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Clip, DataType::Float32,
                ClipCuda, "Clip_CUDA_Float32");

}; // namespace infini

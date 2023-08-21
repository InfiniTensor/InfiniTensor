#include "operators/expand.h"
#include "cuda/cuda_expand.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"

namespace infini {

class ExpandCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ExpandObj>(_op);

        void *const inputData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const outputData = (op->getOutput()->getRawDataPtr<void *>());
        auto shapedim = op->getShape();
        int shape = 1;
        for (size_t i = 0; i < shapedim.size(); ++i)
            shape *= shapedim[i];
        auto dims = op->getInputs(0)->getDims();
        int inputsize = 1;
        for (size_t i = 0; i < dims.size(); ++i)
            inputsize *= dims[i];
        expand_kernel((float *)inputData, (float *)outputData, shape,
                      inputsize);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Expand, DataType::Float32, ExpandCuda,
                "Expand_CUDA_Float32");

}; // namespace infini

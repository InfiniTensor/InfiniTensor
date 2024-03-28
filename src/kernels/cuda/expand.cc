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
        auto a_dim = op->getInputs(0)->getDims();
        auto b_dim = op->getOutput()->getDims(); // output shape

        const int dType = op->getDType().getIndex();
        if (a_dim.size() > 4 || b_dim.size() > 4)
            IT_TODO_HALT();

        int a[4] = {1, 1, 1, 1};
        int b[4] = {1, 1, 1, 1};
        std::copy(a_dim.begin(), a_dim.end(), a + (4 - a_dim.size()));
        std::copy(b_dim.begin(), b_dim.end(), b + (4 - b_dim.size()));
        expandKernel(dType, inputData, outputData, a[0], a[1], a[2], a[3], b[0],
                     b[1], b[2], b[3]);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Expand, ExpandCuda, "Expand_CUDA");

}; // namespace infini

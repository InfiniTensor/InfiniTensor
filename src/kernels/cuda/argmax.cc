#include "operators/argmax.h"
#include "cuda/cuda_argmax.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"

namespace infini {

class ArgMaxCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ArgMaxObj>(_op);
        Tensor input = op->getInputs(0);
        const int dType = input->getDType().getIndex();
        void *const inputData = (op->getInputs(0)->getRawDataPtr<void *>());
        int64_t *const outputData =
            (op->getOutput()->getRawDataPtr<int64_t *>());
        const auto &shapeInput = input->getDims(); // input shape
        auto axis = op->getAxis();
        auto keepDims = op->getKeepDims();
        auto selectLastIndex = op->getSelectLastIndex();
        int outer = 1;
        for (int i = 0; i < axis; ++i) {
            outer *= shapeInput[i];
        }
        int axis_size = shapeInput[axis];
        int inner = 1;
        for (int i = axis + 1; i < (int)shapeInput.size(); ++i) {
            inner *= shapeInput[i];
        }
        argmax_kernel(inputData, outputData, outer, inner, axis_size,
                      selectLastIndex, dType);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::ArgMax, ArgMaxCuda, "ArgMax_CUDA");

} // namespace infini
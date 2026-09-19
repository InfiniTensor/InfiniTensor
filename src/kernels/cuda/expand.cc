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
        const auto &in_Shape = op->getInputs(0)->getDims(); // input shape
        // What comes out, which is not the target: an input dimension wider
        // than the target's keeps its own size, so a target of one over a batch
        // of five yields five.
        const auto &out_Shape = op->getOutput()->getDims(); // output shape

        SmallArray inputShape, outputShape;
        // The kernel walks the output, so it is the output's rank that says how
        // many dimensions there are. A target of a higher rank than the input
        // makes these differ.
        const int nDims = out_Shape.size();
        // Broadcasting lines the two shapes up at the right, so a shorter input
        // sits at the far end of the output and reads as one before it. The
        // kernel pairs the two by index, so the padding has to be written out.
        const int lead = nDims - static_cast<int>(in_Shape.size());

        IT_ASSERT(nDims <= SMALL_ARRAY_SIZE);
        int outputsize = 1; // the length of the output vector after flatten
        for (int i = 0; i < nDims; ++i) {
            outputShape.data[i] = out_Shape[i];
            inputShape.data[i] = i < lead ? 1 : in_Shape[i - lead];
            outputsize *= out_Shape[i];
        }
        const int dType = op->getDType().getIndex();
        expandKernel(dType, inputData, outputData, nDims, outputsize,
                     inputShape, outputShape);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Expand, ExpandCuda, "Expand_CUDA");

}; // namespace infini

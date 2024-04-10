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
        if (a_dim.size() > 4 || b_dim.size() > 4) {
            SmallArray inputShape, outputShape;
            int nDims = op->getInputs(0)->getDims().size();

            IT_ASSERT(nDims <= SMALL_ARRAY_SIZE);
            int outputsize = 1; // the length of the output vector after flatten
            for (int i = 0; i < nDims; ++i) {
                outputShape.data[i] = b_dim[i];
                inputShape.data[i] = a_dim[i];
                outputsize *= b_dim[i];
            }
            const int dType = op->getDType().getIndex();
            expandKernel(dType, inputData, outputData, nDims, outputsize,
                         inputShape, outputShape);

        } else {
            int a[4] = {1, 1, 1, 1};
            int b[4] = {1, 1, 1, 1};
            std::copy(a_dim.begin(), a_dim.end(), a + (4 - a_dim.size()));
            std::copy(b_dim.begin(), b_dim.end(), b + (4 - b_dim.size()));
            expandKernel(dType, inputData, outputData, a[0], a[1], a[2], a[3],
                         b[0], b[1], b[2], b[3]);
        }
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Expand, ExpandCuda, "Expand_CUDA");

}; // namespace infini

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
        const auto &in_Shape = op->getInputs(0)->getDims(); //输入的形状
        const auto &out_Shape = op->getShape(); //扩张的目标形状

        SmallArray inputShape, outputShape;
        int nDims = op->getInputs(0)->getDims().size();
        ; //输入的维度
        IT_ASSERT(nDims <= SMALL_ARRAY_SIZE);
        int outputsize = 1; //计算输出向量flatten以后的长度
        for (int i = 0; i < nDims; ++i) {
            outputShape.data[i] = out_Shape[i];
            inputShape.data[i] = in_Shape[i];
            outputsize *= out_Shape[i];
        }
        expand_kernel((float *)inputData, (float *)outputData, nDims,
                      outputsize, inputShape, outputShape);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Expand, DataType::Float32, ExpandCuda,
                "Expand_CUDA_Float32");

}; // namespace infini

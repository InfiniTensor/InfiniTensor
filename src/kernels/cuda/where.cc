#include "operators/where.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_where.h"

namespace infini {

class WhereCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<WhereObj>(_op);

        void *const inputxData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const inputyData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const conditionData = (op->getInputs(2)->getRawDataPtr<void *>());
        void *const outputData = (op->getOutput()->getRawDataPtr<void *>());
        const auto &inputx_Shape = op->getInputs(0)->getDims();
        const auto &inputy_Shape = op->getInputs(1)->getDims();
        const auto &condition_Shape = op->getInputs(2)->getDims();
        const auto &output_Shape = op->getOutput()->getDims();

        int nDims = op->getInputs(0)->getDims().size();
        IT_ASSERT(nDims <= SMALL_ARRAY_SIZE);

        SmallArray inputxShape, inputyShape, conditionShape, outputShape;
        for (int i = 0; i < nDims; ++i) {
            inputxShape.data[i] = inputx_Shape[i];
            inputyShape.data[i] = inputy_Shape[i];
            conditionShape.data[i] = condition_Shape[i];
            outputShape.data[i] = output_Shape[i];
        }
        where_kernel((float *)inputxData, (float *)inputyData,
                     (float *)conditionData, (float *)outputData, nDims,
                     inputxShape, inputyShape, conditionShape, outputShape);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Where, DataType::Float32, WhereCuda,
                "Where_CUDA_Float32");

}; // namespace infini

#include "operators/where.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_where.h"

namespace infini {

class WhereCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<WhereObj>(_op);

        void *const inputXData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const inputYData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const conditionData = (op->getInputs(2)->getRawDataPtr<void *>());
        void *const outputData = (op->getOutput()->getRawDataPtr<void *>());
        const auto &inputX_Shape = op->getInputs(0)->getDims();
        const auto &inputY_Shape = op->getInputs(1)->getDims();
        const auto &condition_Shape = op->getInputs(2)->getDims();
        const auto &output_Shape = op->getOutput()->getDims();

        const int xSize = op->getInputs(0)->getDims().size();
        const int ySize = op->getInputs(1)->getDims().size();
        const int cSize = op->getInputs(2)->getDims().size();
        int nDims = op->getOutput()->getDims().size();
        IT_ASSERT(nDims <= SMALL_ARRAY_SIZE);

        SmallArray inputXShape, inputYShape, conditionShape, outputShape;
        for (int i = nDims - 1; i >= 0; --i) {
            inputXShape.data[i] = 1;
            inputYShape.data[i] = 1;
            conditionShape.data[i] = 1;
            outputShape.data[i] = output_Shape[i];
        }
        for (int i = xSize - 1; i >= 0; --i) {
            inputXShape.data[i + nDims - xSize] = inputX_Shape[i];
        }
        for (int i = ySize - 1; i >= 0; --i) {
            inputYShape.data[i + nDims - ySize] = inputY_Shape[i];
        }
        for (int i = cSize - 1; i >= 0; --i) {
            conditionShape.data[i + nDims - cSize] = condition_Shape[i];
        }
        where_kernel((float *)inputXData, (float *)inputYData,
                     (int *)conditionData, (float *)outputData, nDims,
                     inputXShape, inputYShape, conditionShape, outputShape);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Where, DataType::Float32, WhereCuda,
                "Where_CUDA_Float32");

}; // namespace infini
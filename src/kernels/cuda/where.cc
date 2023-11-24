#include "operators/where.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_where.h"
#include "utils/broadcast_shape.h"

namespace infini {

class WhereCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<WhereObj>(_op);

        void *const inputXData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const inputYData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const conditionData = (op->getInputs(2)->getRawDataPtr<void *>());
        void *const outputData = (op->getOutput()->getRawDataPtr<void *>());
        const auto &opInputXShape = op->getInputs(0)->getDims();
        const auto &opInputYShape = op->getInputs(1)->getDims();
        const auto &opConditionShape = op->getInputs(2)->getDims();
        const auto &opOutputShape = op->getOutput()->getDims();

        const int xSize = op->getInputs(0)->getRank();
        const int ySize = op->getInputs(1)->getRank();
        const int cSize = op->getInputs(2)->getRank();

        int nDims = op->getOutput()->getDims().size();
        IT_ASSERT(nDims <= SMALL_ARRAY_SIZE);
        int outputsize = 1;
        SmallArray inputXShape, inputYShape, conditionShape, outputShape;
        for (int i = nDims - 1; i >= 0; --i) {
            outputShape.data[i] = opOutputShape[i];
            outputsize *= outputShape.data[i];
        }
        broadcastShape(opInputXShape, inputXShape, nDims, xSize);
        broadcastShape(opInputYShape, inputYShape, nDims, ySize);
        broadcastShape(opConditionShape, conditionShape, nDims, cSize);

        whereKernel((float *)inputXData, (float *)inputYData,
                    (uint8_t *)conditionData, (float *)outputData, nDims,
                    outputsize, inputXShape, inputYShape, conditionShape,
                    outputShape, xSize, ySize, cSize);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Where, DataType::Float32, WhereCuda,
                "Where_CUDA_Float32");

}; // namespace infini

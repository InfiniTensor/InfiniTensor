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

        SmallArray inputXShape, inputYShape, conditionShape, outputShape;
        for (int i = nDims - 1; i >= 0; --i) {
<<<<<<< HEAD
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
=======
            outputShape.data[i] = opOutputShape[i];
        }

        broadcastShape(opInputXShape, inputXShape, nDims, xSize);
        broadcastShape(opInputYShape, inputYShape, nDims, ySize);
        broadcastShape(opConditionShape, conditionShape, nDims, cSize);

        whereKernel((float *)inputXData, (float *)inputYData,
                    (uint8_t *)conditionData, (float *)outputData, nDims,
                    inputXShape, inputYShape, conditionShape, outputShape);
>>>>>>> ec3adf6fa73cc6390f09a9bbd23910640d9ed000
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Where, DataType::Float32, WhereCuda,
                "Where_CUDA_Float32");

}; // namespace infini

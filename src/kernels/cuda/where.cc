#include "operators/where.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
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

        auto a_dim = op->getInputs(0)->getDims();
        auto b_dim = op->getInputs(1)->getDims();
        auto c_dim = op->getInputs(2)->getDims();
        auto d_dim = op->getOutput()->getDims();
        const int dTypeIndex = op->getDType().getIndex();
        if (a_dim.size() > 4 || b_dim.size() > 4 || c_dim.size() > 4 ||
            d_dim.size() > 4) {
            const int xSize = op->getInputs(0)->getRank();
            const int ySize = op->getInputs(1)->getRank();
            const int cSize = op->getInputs(2)->getRank();

            int nDims = op->getOutput()->getDims().size();
            IT_ASSERT(nDims <= SMALL_ARRAY_SIZE);
            int outputsize = 1;
            SmallArray inputXShape, inputYShape, conditionShape, outputShape;
            for (int i = nDims - 1; i >= 0; --i) {
                outputShape.data[i] = d_dim[i];
                outputsize *= outputShape.data[i];
            }
            broadcastShape(a_dim, inputXShape, nDims, xSize);
            broadcastShape(b_dim, inputYShape, nDims, ySize);
            broadcastShape(c_dim, conditionShape, nDims, cSize);
            whereKernel(dTypeIndex, inputXData, inputYData,
                        (uint8_t *)conditionData, outputData, nDims, outputsize,
                        inputXShape, inputYShape, conditionShape, outputShape,
                        xSize, ySize, cSize);
        }

        else {
            int a[4] = {1, 1, 1, 1};
            int b[4] = {1, 1, 1, 1};
            int c[4] = {1, 1, 1, 1};
            int d[4] = {1, 1, 1, 1};

            std::copy(a_dim.begin(), a_dim.end(), a + (4 - a_dim.size()));
            std::copy(b_dim.begin(), b_dim.end(), b + (4 - b_dim.size()));
            std::copy(c_dim.begin(), c_dim.end(), c + (4 - c_dim.size()));
            std::copy(d_dim.begin(), d_dim.end(), d + (4 - d_dim.size()));

            whereKernel(dTypeIndex, inputXData, inputYData,
                        (uint8_t *)conditionData, outputData, a[0], a[1], a[2],
                        a[3], b[0], b[1], b[2], b[3], c[0], c[1], c[2], c[3],
                        d[0], d[1], d[2], d[3]);
        }
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Where, WhereCuda, "Where_CUDA");

}; // namespace infini

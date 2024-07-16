#include "operators/range.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_range.h"
#include "utils/operator_utils.h"

namespace infini {

class RangeCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<RangeObj>(_op);

        auto startData = op->getStart();
        auto limitData = op->getLimit();
        auto deltaData = op->getDelta();

        void *const outputData = (op->getOutput()->getRawDataPtr<void *>());
        IT_ASSERT(startData < limitData);
        //IT_ASSERT(op->getDType() == DataType::Float32);
        int size = std::max(std::ceil((limitData - startData) / deltaData), 0.0f);


        range_kernel(startData, limitData, deltaData, (float *)outputData, size
                    );
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Range, RangeCuda,
                "Range_CUDA_Int32");

}; // namespace infini
#include "operators/where.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_where.h"

namespace infini {

class WhereCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<WhereObj>(_op);

        void *const inputData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const otherData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const conditionData = (op->getInputs(2)->getRawDataPtr<void *>());
        void *const outputData = (op->getOutput()->getRawDataPtr<void *>());
        auto dims = op->getInputs(0)->getDims();

        int size = 1;
        for (size_t i = 0; i < dims.size(); ++i)
            size *= dims[i];
        where_kernel((float *)inputData, (float *)otherData,
                     (float *)conditionData, (float *)outputData, size);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Where, DataType::Float32, WhereCuda,
                "Where_CUDA_Float32");

}; // namespace infini

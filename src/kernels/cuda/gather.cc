#include "operators/gather.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"
#include "cuda/gather.h"

namespace infini {
class GatherCuda : public CudaKernelWithoutConfig {

    void compute(const Operator &op,
                 const RuntimeObj *_context) const override {

        auto input = op->getInputs(0);
        auto index = op->getInputs(1);

        GatherMetaData metaData;
        initGatherMetaData(metaData, op);

        void *const inputData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const outputData = (op->getOutput()->getRawDataPtr<void *>());

        if (op->getDType() == DataType::Float32) {
            gather_kernel<float>((float *)inputData, (float *)outputData,
                                 metaData, op->getOutput()->size());
        } else if (op->getDType() == DataType::Float16) {
            gather_kernel<half>((half *)inputData, (half *)outputData, metaData,
                                op->getOutput()->size());
        }
        else if (op->getDType() == DataType::Int8) {
            gather_kernel<int8_t>((int8_t *)inputData, (int8_t *)outputData, metaData,
                                op->getOutput()->size());
        }
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Gather, GatherCuda, "Gather_CUDA");
} // namespace infini

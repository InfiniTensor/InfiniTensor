#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"
#include "cuda/gather.h"
#include "operators/gather.h"

namespace infini {

class GatherElementsCuda : public CudaKernelWithoutConfig {

    void compute(const Operator &op,
                 const RuntimeObj *_context) const override {

        GatherMetaData metaData;
        initGatherMetaData(metaData, op);

        auto input = op->getInputs(0);
        auto index = op->getInputs(1);
        auto output = op->getOutput();
        void *inData = input->getRawDataPtr<void *>();
        void *outData = output->getRawDataPtr<void *>();
        gather_elements_kernel(inData, outData, metaData,
                               op->getOutput()->size());
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::GatherElements, DataType::Float32,
                GatherElementsCuda, "GatherELements_CUDA_Float32");
REGISTER_KERNEL(Device::CUDA, OpType::GatherElements, DataType::Int32,
                GatherElementsCuda, "GatherElements_CUDA_Int32");
} // namespace infini

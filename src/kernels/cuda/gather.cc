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

        auto inData = input->getRawDataPtr<float *>();
        auto outData = op->getOutput()->getRawDataPtr<float *>();
        gather_kernel(inData, outData, metaData, op->getOutput()->size());
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Gather, GatherCuda, "Gather_CUDA");
} // namespace infini

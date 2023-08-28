#ifdef INFINI_USE_NCCL
#include "operators/all_gather.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"
#include "cuda/nccl_communicator.h"

namespace infini {
class AllGatherNCCL : public CudaKernelWithoutConfig {
  public:
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<AllGatherObj>(_op);
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);
        int world_size = op->getWorldSize();
        // Check if world size info in operator matches runtime
        IT_ASSERT(world_size == context->getCommunicator().getWorldSize());

        void *input = op->getInputs(0)->getRawDataPtr<void *>();
        CudaPtr output_temp =
            context->getWorkspace(op->getInputs(0)->getBytes() * world_size);
        // void *output = op->getOutput()->getRawDataPtr<void *>();
        IT_ASSERT(op->getDType() == DataType::Float32);
        size_t bytes = op->getInputs(0)->getBytes();
        size_t count = bytes / op->getDType().getSize();

        ncclComm_t comm =
            dynamic_cast<NcclCommunicatorObj &>(context->getCommunicator())
                .getNcclComm();
        // TODO: Using default stream 0 for now.
        checkNcclError(
            ncclAllGather(input, output_temp, count, ncclFloat, comm, 0));

        for (int i = 0; i < world_size; ++i) {
            Tensor output = op->getOutput(i);
            context->copyBlobInsideRuntime(
                output->getRawDataPtr<float *>(),
                static_cast<float *>(output_temp) + i * count, bytes);
        }
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::AllGather, DataType::Float32,
                AllGatherNCCL, "AllGather_NCCL_CUDA_Float32");
} // namespace infini

#endif

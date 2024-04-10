#ifdef INFINI_USE_HCCL
#include "operators/all_gather.h"
#include "ascend/ascend_kernel_without_config.h"
#include "ascend/ascend_runtime.h"
#include "ascend/hccl_communicator.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

namespace infini {
class AllGatherHCCL : public ASCENDKernelWithoutConfig {
  public:
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<AllGatherObj>(_op);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);
        int world_size = op->getWorldSize();
        // Check if world size info in operator matches runtime
        IT_ASSERT(world_size == context->getCommunicator().getWorldSize());

        void *input = op->getInputs(0)->getRawDataPtr<void *>();
        ASCENDPtr output_temp =
            context->getWorkspace(op->getInputs(0)->getBytes() * world_size);
        // void *output = op->getOutput()->getRawDataPtr<void *>();
        IT_ASSERT(op->getDType() == DataType::Float32);
        size_t bytes = op->getInputs(0)->getBytes();
        size_t count = bytes / op->getDType().getSize();

        HcclComm comm =
            dynamic_cast<HcclCommunicatorObj &>(context->getCommunicator())
                .getHcclComm();
        // TODO: Using default stream 0 for now.
        HCCLCHECK(HcclAllGather(input, output_temp, uint64_t(count),
                                HCCL_DATA_TYPE_FP32, comm,
                                context->ASCENDHandle()));
        ACLCHECK(aclrtSynchronizeStream(context->ASCENDHandle()));

        for (int i = 0; i < world_size; ++i) {
            Tensor output = op->getOutput(i);
            context->copyBlobInsideRuntime(
                output->getRawDataPtr<float *>(),
                static_cast<float *>(output_temp) + i * count, bytes);
        }
    }
};

REGISTER_KERNEL(Device::ASCEND, OpType::AllGather, AllGatherHCCL,
                "AllGather_HCCL_ASCEND");
} // namespace infini

#endif

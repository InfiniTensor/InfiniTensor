#ifdef INFINI_USE_CNCL
#include "operators/all_gather.h"
#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"
#include "bang/cncl_communicator.h"
#include <thread>
namespace infini {
class AllGatherCNCL : public BangKernelWithoutConfig {
  public:
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<AllGatherObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);
        int world_size = op->getWorldSize();
        // Check if world size info in operator matches runtime
        IT_ASSERT(world_size == context->getCommunicator().getWorldSize());

        void *input = op->getInputs(0)->getRawDataPtr<void *>();
        BangPtr output_temp =
            context->getWorkspace(op->getInputs(0)->getBytes() * world_size);
        // void *output = op->getOutput()->getRawDataPtr<void *>();
        IT_ASSERT(op->getDType() == DataType::Float32);
        size_t bytes = op->getInputs(0)->getBytes();
        size_t count = bytes / op->getDType().getSize();

        cnclComm_t comm =
            dynamic_cast<CnclCommunicatorObj &>(context->getCommunicator())
                .getCnclComm();
        cnrtQueue_t queue =
            dynamic_cast<CnclCommunicatorObj &>(context->getCommunicator())
                .getCnclQueue();
        // TODO: Using default stream 0 for now.
        CNCL_CHECK(
            cnclAllGather(input, output_temp, count, cnclFloat, comm, queue));
        checkBangError(cnrtQueueSync(queue));
        for (int i = 0; i < world_size; ++i) {
            Tensor output = op->getOutput(i);
            context->copyBlobInsideRuntime(
                output->getRawDataPtr<float *>(),
                static_cast<float *>(output_temp) + i * count, bytes);
        }
    }
};

REGISTER_KERNEL(Device::BANG, OpType::AllGather, DataType::Float32,
                AllGatherCNCL, "AllGather_CNCL_BANG_Float32");
} // namespace infini

#endif

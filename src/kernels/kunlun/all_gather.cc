#ifdef INFINI_USE_XCCL
#include "operators/all_gather.h"
#include "kunlun/kunlun_kernel_without_config.h"
#include "kunlun/kunlun_runtime.h"
#include "kunlun/xccl_communicator.h"

namespace infini {
class AllGatherXCCL : public KUNLUNKernelWithoutConfig {
  public:
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<AllGatherObj>(_op);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);
        int world_size = op->getWorldSize();
        IT_ASSERT(world_size == context->getCommunicator().getWorldSize());
        void *input = op->getInputs(0)->getRawDataPtr<void *>();
        KUNLUNPtr output_temp =
            context->getWorkspace(op->getInputs(0)->getBytes() * world_size);
        IT_ASSERT(op->getDType() == DataType::Float32);
        size_t bytes = op->getInputs(0)->getBytes();
        size_t count = bytes / op->getDType().getSize();

        BKCLContext_t comm =
            dynamic_cast<XcclCommunicatorObj &>(context->getCommunicator())
                .getXcclComm();
        // TODO: Using the default stream 0
        checkXcclError(
            bkcl_all_gather(comm, input, count, output_temp, BKCL_FLOAT, 0));

        for (int i = 0; i < world_size; ++i) {
            Tensor output = op->getOutput(i);
            context->copyBlobInsideRuntime(
                output->getRawDataPtr<float *>(),
                static_cast<float *>(output_temp) + i * count, bytes);
        }
    }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::AllGather, DataType::Float32,
                AllGatherXCCL, "AllGatcher_XCCL_KUNLUN_Float32");
} // namespace infini
#endif

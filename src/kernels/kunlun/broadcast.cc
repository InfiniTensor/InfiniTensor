#ifdef INFINI_USE_XCCL
#include "operators/broadcast.h"
#include "kunlun/kunlun_kernel_without_config.h"
#include "kunlun/kunlun_runtime.h"
#include "kunlun/xccl_communicator.h"

namespace infini {
class BroadcastXCCL : public KUNLUNKernelWithoutConfig {
  public:
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<BroadcastObj>(_op);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);
        void *input = op->getInputs(0)->getRawDataPtr<void *>();
        void *output = op->getOutput()->getRawDataPtr<void *>();
        IT_ASSERT(op->getDType() == DataType::Float32);
        size_t count = op->getInputs(0)->getBytes() / op->getDType().getSize();

        BKCLContext_t comm =
            dynamic_cast<XcclCommunicatorObj &>(context->getCommunicator())
                .getXcclComm();
        // TODO: Using default stream 0 for now.
        checkXcclError(bkcl_broadcast(comm, input, output, count, BKCL_FLOAT, op->getRoot(), 0));
    }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::Broadcast, DataType::Float32,
                BroadcastXCCL, "Broadcast_XCCL_KUNLUN_Float32");
} // namespace infini
#endif

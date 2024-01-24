#ifdef INFINI_USE_CNCL
#include "operators/broadcast.h"
#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"
#include "bang/cncl_communicator.h"
#include <thread>
namespace infini {
class BroadcastCNCL : public BangKernelWithoutConfig {
  public:
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<BroadcastObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);
        void *input = op->getInputs(0)->getRawDataPtr<void *>();
        void *output = op->getOutput()->getRawDataPtr<void *>();
        size_t bytes = op->getInputs(0)->getBytes();
        size_t count = bytes / sizeof(uint8_t);

        cnclComm_t comm =
            dynamic_cast<CnclCommunicatorObj &>(context->getCommunicator())
                .getCnclComm();
        cnrtQueue_t queue = context->getBangQueue();
        // TODO: Using default stream 0 for now.
        CNCL_CHECK(cnclBroadcast(input, output, count, cnclUint8,
                                 op->getRoot(), comm, queue));
        checkBangError(cnrtQueueSync(queue));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::Broadcast,
                BroadcastCNCL, "Broadcast_CNCL_BANG");
} // namespace infini

#endif

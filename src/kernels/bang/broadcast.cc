#ifdef INFINI_USE_CNCL
#include "operators/broadcast.h"
#include "bang/bang_kernel_wihtout_config.h"
#include "bang/bang_runtime.h"
#include "bang/cnnl_communicator.h"

namespace infini {
class BroadcastCNCL : public BangKernelWithoutConfig {
  public:
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<BroadcastObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);
        void *input = op->getInputs(0)->getRawDataPtr<void *>();
        void *output = op->getOutput()->getRawDataPtr<void *>();
        IT_ASSERT(op->getDType() == DataType::Float32);
        size_t count = op->getInputs(0)->getBytes() / op->getDType().getSize();

        cnnlComm_t comm =
            dynamic_cast<CnclCommunicatorObj &>(context->getCommunicator())
                .getCnclComm();
        // TODO: Using default stream 0 for now.
        checkCnclError(cnnlBroadcast(input, output, count, cnnlFloat,
                                     op->getRoot(), comm, 0));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::Broadcast, DataType::Float32,
                BroadcastCNCL, "Broadcast_CNCL_BANG_Float32");
} // namespace infini

#endif

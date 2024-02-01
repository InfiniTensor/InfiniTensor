#ifdef INFINI_USE_CNCL
#include "operators/all_reduce.h"
#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"
#include "bang/cncl_communicator.h"
#include <thread>
namespace infini {
class AllReduceCNCL : public BangKernelWithoutConfig {
  public:
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<AllReduceBaseObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);
        void *input = op->getInputs(0)->getRawDataPtr<void *>();
        void *output = op->getOutput()->getRawDataPtr<void *>();
        size_t bytes = op->getInputs(0)->getBytes();
        size_t count = bytes / op->getDType().getSize();
        cnclComm_t comm =
            dynamic_cast<CnclCommunicatorObj &>(context->getCommunicator())
                .getCnclComm();
        cnrtQueue_t queue = context->getBangQueue();
        // checkBangError(cnrtQueueSync(queue));
        CNCL_CHECK(cnclAllReduce(input, output, count,
                                 cnclDataTypeConvert(op->getDType()),
                                 getRedOp(), comm, queue));
        checkBangError(cnrtQueueSync(queue));
    }

    virtual cnclReduceOp_t getRedOp() const = 0;
};

class AllReduceSumCNCL : public AllReduceCNCL {
    cnclReduceOp_t getRedOp() const override { return cnclSum; }
};
class AllReduceProdCNCL : public AllReduceCNCL {
    cnclReduceOp_t getRedOp() const override { return cnclProd; }
};
class AllReduceMinCNCL : public AllReduceCNCL {
    cnclReduceOp_t getRedOp() const override { return cnclMin; }
};
class AllReduceMaxCNCL : public AllReduceCNCL {
    cnclReduceOp_t getRedOp() const override { return cnclMax; }
};

REGISTER_KERNEL(Device::BANG, OpType::AllReduceSum, AllReduceSumCNCL,
                "AllReduce_Sum_CNCL_BANG");
REGISTER_KERNEL(Device::BANG, OpType::AllReduceProd, AllReduceProdCNCL,
                "AllReduce_Prod_CNCL_BANG");
REGISTER_KERNEL(Device::BANG, OpType::AllReduceMin, AllReduceMinCNCL,
                "AllReduce_Min_CNCL_BANG");
REGISTER_KERNEL(Device::BANG, OpType::AllReduceMax, AllReduceMaxCNCL,
                "AllReduce_Max_CNCL_BANG");
} // namespace infini
#endif

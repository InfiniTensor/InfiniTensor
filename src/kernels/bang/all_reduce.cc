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
        IT_ASSERT(op->getDType() == DataType::Float32);
        size_t count = op->getInputs(0)->size();
        cnclComm_t comm =
            dynamic_cast<CnclCommunicatorObj &>(context->getCommunicator())
                .getCnclComm();
        cnrtQueue_t queue = context->getBangQueue();
        // checkBangError(cnrtQueueSync(queue));
        CNCL_CHECK(cnclAllReduce(input, output, count, cnclFloat, getRedOp(),
                                 comm, queue));
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

REGISTER_KERNEL(Device::BANG, OpType::AllReduceSum, DataType::Float32,
                AllReduceSumCNCL, "AllReduce_Sum_CNCL_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::AllReduceProd, DataType::Float32,
                AllReduceProdCNCL, "AllReduce_Prod_CNCL_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::AllReduceMin, DataType::Float32,
                AllReduceMinCNCL, "AllReduce_Min_CNCL_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::AllReduceMax, DataType::Float32,
                AllReduceMaxCNCL, "AllReduce_Max_CNCL_BANG_Float32");
} // namespace infini
#endif

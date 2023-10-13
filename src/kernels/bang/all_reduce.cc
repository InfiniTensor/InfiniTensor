#ifdef INFINI_USE_CNCL
#include "operators/all_reduce.h"
#include "bang/bang_kernel_wihtout_config.h"
#include "bang/bang_runtime.h"
#include "bang/cncl_communicator.h"

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
        // TODO: Using default stream 0 for now.
        checkCnclError(cnclAllReduce(input, output, count, cnclFloat32,//checkCnclError函数在bang/cncl_communicator.h定义
                                     getRedOp(), comm, 0));//queues[i] = 0 ?
    }

    virtual cnclRedOp_t getRedOp() const = 0;
};

class AllReduceSumCNCL : public AllReduceCNCL {
    cnclRedOp_t getRedOp() const override { return cnclSum; }
};
class AllReduceProdCNCL : public AllReduceCNCL {
    cnclRedOp_t getRedOp() const override { return cnclProd; }
};
class AllReduceMinCNCL : public AllReduceCNCL {
    cnclRedOp_t getRedOp() const override { return cnclMin; }
};
class AllReduceMaxCNCL : public AllReduceCNCL {
    cnclRedOp_t getRedOp() const override { return cnclMax; }
};
class AllReduceAvgCNCL : public AllReduceCNCL {
    cnclRedOp_t getRedOp() const override { return cnclAvg; }
};

REGISTER_KERNEL(Device::BANG, OpType::AllReduceSum, DataType::Float32,
                AllReduceSumCNCL, "AllReduce_Sum_CNCL_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::AllReduceProd, DataType::Float32,
                AllReduceProdCNCL, "AllReduce_Prod_CNCL_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::AllReduceMin, DataType::Float32,
                AllReduceMinCNCL, "AllReduce_Min_CNCL_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::AllReduceMax, DataType::Float32,
                AllReduceMaxCNCL, "AllReduce_Max_CNCL_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::AllReduceAvg, DataType::Float32,
                AllReduceAvgCNCL, "AllReduce_Avg_CNCL_BANG_Float32");
} // namespace infini
#endif

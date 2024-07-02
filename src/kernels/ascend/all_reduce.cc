#ifdef INFINI_USE_HCCL
#include "operators/all_reduce.h"
#include "ascend/ascend_kernel_without_config.h"
#include "ascend/ascend_runtime.h"
#include "ascend/hccl_communicator.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

namespace infini {
class AllReduceHCCL : public ASCENDKernelWithoutConfig {
  public:
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<AllReduceBaseObj>(_op);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);
        void *sendBuff = op->getInputs(0)->getRawDataPtr<void *>();
        void *recvBuff = op->getOutput()->getRawDataPtr<void *>();

        // HcclDataType

        size_t count = op->getInputs(0)->size();

        HcclComm comm =
            dynamic_cast<HcclCommunicatorObj &>(context->getCommunicator())
                .getHcclComm();
        // TODO: Using default stream 0 for now.
        HCCLCHECK(HcclAllReduce(sendBuff, recvBuff, count, HCCL_DATA_TYPE_FP32,
                                getRedOp(), comm, context->ASCENDHandle()));
        ACLCHECK(aclrtSynchronizeStream(context->ASCENDHandle()));
    }

    virtual HcclReduceOp getRedOp() const = 0;
};

class AllReduceSumHCCL : public AllReduceHCCL {
    HcclReduceOp getRedOp() const override { return HCCL_REDUCE_SUM; }
};
class AllReduceProdHCCL : public AllReduceHCCL {
    HcclReduceOp getRedOp() const override { return HCCL_REDUCE_PROD; }
};
class AllReduceMinHCCL : public AllReduceHCCL {
    HcclReduceOp getRedOp() const override { return HCCL_REDUCE_MIN; }
};
class AllReduceMaxHCCL : public AllReduceHCCL {
    HcclReduceOp getRedOp() const override { return HCCL_REDUCE_MAX; }
};

REGISTER_KERNEL(Device::ASCEND, OpType::AllReduceSum, AllReduceSumHCCL,
                "AllReduce_Sum_HCCL_ASCEND");
REGISTER_KERNEL(Device::ASCEND, OpType::AllReduceProd, AllReduceProdHCCL,
                "AllReduce_Prod_HCCL_ASCEND");
REGISTER_KERNEL(Device::ASCEND, OpType::AllReduceMin, AllReduceMinHCCL,
                "AllReduce_Min_HCCL_ASCEND");
REGISTER_KERNEL(Device::ASCEND, OpType::AllReduceMax, AllReduceMaxHCCL,
                "AllReduce_Max_HCCL_ASCEND");

} // namespace infini
#endif

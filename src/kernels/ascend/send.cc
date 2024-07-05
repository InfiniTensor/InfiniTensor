#ifdef INFINI_USE_HCCL
#include "operators/send.h"
#include "ascend/ascend_kernel_without_config.h"
#include "ascend/ascend_runtime.h"
#include "ascend/hccl_communicator.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

namespace infini {
class SendHCCL : public ASCENDKernelWithoutConfig {
  public:
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<SendObj>(_op);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);
        void *input = op->getInputs(0)->getRawDataPtr<void *>();

        IT_ASSERT(op->getDType() == DataType::Float32);
        int inputCount =
            op->getInputs(0)->getBytes() / op->getDType().getSize();

        HcclComm comm =
            dynamic_cast<HcclCommunicatorObj &>(context->getCommunicator())
                .getHcclComm();

        uint32_t rank;

        checkHCCLError(HcclGetRankId(comm, &rank));

        int source = op->getSourceRank();
        int destination = op->getDestinationRank();

        if (int(rank) == source) {
            checkHCCLError(HcclSend(input, uint64_t(inputCount),
                                    HCCL_DATA_TYPE_FP32, uint32_t(destination),
                                    comm, context->ASCENDHandle()));
        }
        checkASCENDError(aclrtSynchronizeStream(context->ASCENDHandle()));
    }
};

REGISTER_KERNEL(Device::ASCEND, OpType::Send, SendHCCL, "Send_HCCL_ASCEND");
} // namespace infini

#endif

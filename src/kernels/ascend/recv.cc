#ifdef INFINI_USE_HCCL
#include "operators/recv.h"
#include "ascend/ascend_kernel_without_config.h"
#include "ascend/ascend_runtime.h"
#include "ascend/hccl_communicator.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

namespace infini {
class RecvHCCL : public ASCENDKernelWithoutConfig {
  public:
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<RecvObj>(_op);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);

        void *output = op->getOutput(0)->getRawDataPtr<void *>();
        IT_ASSERT(op->getDType() == DataType::Float32);
        const auto shape = op->getShape();
        int nDims = shape.size();
        int outputCount = 1;
        for (int i = 0; i < nDims; i++) {
            outputCount *= shape[i];
        }

        HcclComm comm =
            dynamic_cast<HcclCommunicatorObj &>(context->getCommunicator())
                .getHcclComm();
        // TODO: Using default stream 0 for now.
        uint32_t rank;

        HCCLCHECK(HcclGetRankId(comm, &rank));

        int source = op->getSourceRank();
        int destination = op->getDestinationRank();

        // printf("###rank:%u,source:%d,outputCount:%d,destination:%d\n", rank,
        //        source, outputCount, destination);
        if (int(rank) == destination) {
            HCCLCHECK(HcclRecv(output, uint64_t(outputCount),
                               HCCL_DATA_TYPE_FP32, uint32_t(source), comm,
                               context->ASCENDHandle()));
        }
        ACLCHECK(aclrtSynchronizeStream(context->ASCENDHandle()));
    }
};

REGISTER_KERNEL(Device::ASCEND, OpType::Recv, RecvHCCL, "Recv_HCCL_ASCEND");
} // namespace infini

#endif

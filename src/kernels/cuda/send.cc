#ifdef INFINI_USE_NCCL
#include "operators/send.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"
#include "cuda/nccl_communicator.h"

namespace infini {
class SendNCCL : public CudaKernelWithoutConfig {
  public:
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<SendObj>(_op);
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);
        void *input = op->getInputs(0)->getRawDataPtr<void *>();

        IT_ASSERT(op->getDType() == DataType::Float32);
        size_t inputCount =
            op->getInputs(0)->getBytes() / op->getDType().getSize();

        ncclComm_t comm =
            dynamic_cast<NcclCommunicatorObj &>(context->getCommunicator())
                .getNcclComm();
        // TODO: Using default stream 0 for now.
        int rank;

        checkNcclError(ncclCommUserRank(comm, &rank));

        int source = op->getSourceRank();
        int destination = op->getDestinationRank();

        if (rank == source) {

            checkNcclError(
                ncclSend(input, inputCount, ncclFloat, destination, comm, 0));
        }
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Send, SendNCCL, "Send_NCCL_CUDA");
} // namespace infini

#endif

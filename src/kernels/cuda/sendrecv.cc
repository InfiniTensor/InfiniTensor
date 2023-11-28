#ifdef INFINI_USE_NCCL
#include "operators/sendrecv.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"
#include "cuda/nccl_communicator.h"

namespace infini {
class SendRecvNCCL : public CudaKernelWithoutConfig {
  public:
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<SendRecvObj>(_op);
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);
        void *input = op->getInputs(0)->getRawDataPtr<void *>();
        void *output = op->getOutput(0)->getRawDataPtr<void *>();
        IT_ASSERT(op->getDType() == DataType::Float32);
        size_t inputCount =
            op->getInputs(0)->getBytes() / op->getDType().getSize();
        const auto shape = op->getShape();
        int nDims = shape.size();
        int outputCount = 1;
        for (int i = 0; i < nDims; i++) {
            outputCount *= shape[i];
        }

        ncclComm_t comm =
            dynamic_cast<NcclCommunicatorObj &>(context->getCommunicator())
                .getNcclComm();
        // TODO: Using default stream 0 for now.
        int rank;

        checkNcclError(ncclCommUserRank(comm, &rank));
        // std::cout << rank << "," << count << std::endl;
        int source = op->getSource();
        int destination = op->getDestination();

        if (rank == source) {

            checkNcclError(
                ncclSend(input, inputCount, ncclFloat, destination, comm, 0));
        }
        if (rank == destination) {

            checkNcclError(
                ncclRecv(output, outputCount, ncclFloat, source, comm, 0));
        }
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::SendRecv, DataType::Float32, SendRecvNCCL,
                "SendRecv_NCCL_CUDA_Float32");
} // namespace infini

#endif

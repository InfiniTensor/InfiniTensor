#ifdef INFINI_USE_NCCL
#include "operators/all_reduce.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"
#include "cuda/nccl_communicator.h"

namespace infini {
class AllReduceNCCL : public CudaKernelWithoutConfig {
  public:
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<AllReduceBaseObj>(_op);
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);
        void *input = op->getInputs(0)->getRawDataPtr<void *>();
        void *output = op->getOutput()->getRawDataPtr<void *>();
        ncclDataType_t ncclType = ncclFloat;
        if (op->getDType() == DataType::Float16) {
            ncclType = ncclFloat16;
        } else if (op->getDType() == DataType::Int8) {
            ncclType = ncclInt8;
        }
        size_t count = op->getInputs(0)->size();

        ncclComm_t comm =
            dynamic_cast<NcclCommunicatorObj &>(context->getCommunicator())
                .getNcclComm();
        // TODO: Using default stream 0 for now.
        checkNcclError(
            ncclAllReduce(input, output, count, ncclType, getRedOp(), comm, 0));
        // checkNcclError(ncclAllReduce(input, output, count, ncclType,
        // getRedOp(),
        //                              comm, CUDAStream::stream));
    }

    virtual ncclRedOp_t getRedOp() const = 0;
};

class AllReduceSumNCCL : public AllReduceNCCL {
    ncclRedOp_t getRedOp() const override { return ncclSum; }
};
class AllReduceProdNCCL : public AllReduceNCCL {
    ncclRedOp_t getRedOp() const override { return ncclProd; }
};
class AllReduceMinNCCL : public AllReduceNCCL {
    ncclRedOp_t getRedOp() const override { return ncclMin; }
};
class AllReduceMaxNCCL : public AllReduceNCCL {
    ncclRedOp_t getRedOp() const override { return ncclMax; }
};
class AllReduceAvgNCCL : public AllReduceNCCL {
    ncclRedOp_t getRedOp() const override { return ncclAvg; }
};

REGISTER_KERNEL(Device::CUDA, OpType::AllReduceSum, AllReduceSumNCCL,
                "AllReduce_Sum_NCCL_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::AllReduceProd, AllReduceProdNCCL,
                "AllReduce_Prod_NCCL_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::AllReduceMin, AllReduceMinNCCL,
                "AllReduce_Min_NCCL_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::AllReduceMax, AllReduceMaxNCCL,
                "AllReduce_Max_NCCL_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::AllReduceAvg, AllReduceAvgNCCL,
                "AllReduce_Avg_NCCL_CUDA");

} // namespace infini
#endif

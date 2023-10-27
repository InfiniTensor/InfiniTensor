#ifdef INFINI_USE_XCCL
#include "operators/all_reduce.h"
#include "kunlun/kunlun_kernel_without_config.h"
#include "kunlun/kunlun_runtime.h"
#include "kunlun/xccl_communicator.h"

namespace infini {
class AllReduceXCCL : public KUNLUNKernelWithoutConfig {
  public:
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<AllReduceBaseObj>(_op);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);
        void *input = op->getInputs(0)->getRawDataPtr<void *>();
        void *output = op->getOutput(0)->getRawDataPtr<void *>();
        IT_ASSERT(op->getDType() == DataType::Float32);
        size_t count = op->getInputs(0)->size();

        BKCLContext_t comm =
            dynamic_cast<XcclCommunicatorObj &>(context->getCommunicator())
                .getXcclComm();
        checkXcclError(bkcl_all_reduce(comm, input, output, count,
                                       BKCLDataType::BKCL_FLOAT, getRedOp(),
                                       0));
        /**
         * The XCCL interface is asynchronous by default.
         * The host returns immediately after the call is completed.
         * Therefore, the calculation results need to be synchronized
         * by xpu_wait(comm_stream)
         */
        xpu_wait(0);
    }
    virtual BKCLOp getRedOp() const = 0;
};

class AllReduceSumXCCL : public AllReduceXCCL {
    BKCLOp getRedOp() const override { return BKCLOp::BKCL_ADD; }
};

class AllReduceMinXCCL : public AllReduceXCCL {
    BKCLOp getRedOp() const override { return BKCLOp::BKCL_MIN; }
};

class AllReduceMaxXCCL : public AllReduceXCCL {
    BKCLOp getRedOp() const override { return BKCLOp::BKCL_MAX; }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::AllReduceSum, DataType::Float32,
                AllReduceSumXCCL, "AllReduce_Sum_XCCL_KUNLUN_FLOAT32");
REGISTER_KERNEL(Device::KUNLUN, OpType::AllReduceMax, DataType::Float32,
                AllReduceMaxXCCL, "AllReduce_Max_XCCL_KUNLUN_FLOAT32");
REGISTER_KERNEL(Device::KUNLUN, OpType::AllReduceMin, DataType::Float32,
                AllReduceMinXCCL, "AllReduce_Min_XCCL_KUNLUN_FLOAT32");
} // namespace infini
#endif

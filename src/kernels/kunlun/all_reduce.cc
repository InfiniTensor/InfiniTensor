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
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);
        void *input = op->getInputs(0)->getRawDataPtr<void *>();
        void *output = op->getOutput(0)->getRawDataPtr<void *>();
        IT_ASSERT(op->getDType() == DataType::Float32);
        size_t count = op->getInputs(0)->size();

        BKCLContext_t comm =
            dynamic_cast<XcclCommunicatorObj &>(context->getCommunicator())
                .getXcclComm();
        // double t = timeit(
        // [&]() {
        checkXcclError(bkcl_all_reduce(comm, input, output, count,
                                       BKCLDataType::BKCL_FLOAT, getRedOp(),
                                       0));
        // },
        // [&]() { context->sync(); });
        // std::cout << "Time consuming for " << op->getInputs(0)->size() << "
        // size is " << t << std::endl;
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

REGISTER_KERNEL(Device::KUNLUN, OpType::AllReduceSum, AllReduceSumXCCL,
                "AllReduce_Sum_XCCL_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::AllReduceMax, AllReduceMaxXCCL,
                "AllReduce_Max_XCCL_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::AllReduceMin, AllReduceMinXCCL,
                "AllReduce_Min_XCCL_KUNLUN");
} // namespace infini
#endif

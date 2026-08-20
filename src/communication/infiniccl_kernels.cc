#ifdef USE_INFINICCL
#include "communication/infiniccl_communicator.h"
#include "core/infini_runtime.h"
#include "core/kernel.h"
#include "operators/all_gather.h"
#include "operators/all_reduce.h"
#include "operators/broadcast.h"
#include "operators/recv.h"
#include "operators/send.h"

namespace infini {
namespace {

class CommunicationKernel : public Kernel {
  public:
    void compute(const Operator &op, const PerfRecord &,
                 const RuntimeObj *context) const override {
        compute(op, context);
    }

    PerfRecord tune(const Operator &op,
                    const RuntimeObj *context) const override {
        return make_ref<PerfRecordObj>(timeit([&]() { compute(op, context); }));
    }

    virtual void compute(const Operator &op,
                         const RuntimeObj *context) const = 0;
};

const InfiniCclCommunicatorObj &getInfiniCcl(const RuntimeObj *context) {
    return dynamic_cast<const InfiniCclCommunicatorObj &>(
        context->getCommunicator());
}

void *getInfiniStream(const RuntimeObj *context) {
    auto runtime = dynamic_cast<const InfiniRuntimeObj *>(context);
    IT_ASSERT(runtime != nullptr, "InfiniCCL kernels require an InfiniRuntime");
    return runtime->getStream();
}

infinicclRedOp_t toInfiniCclReduction(OpType opType) {
    switch (opType.underlying()) {
    case OpType::AllReduceSum:
        return infinicclSum;
    case OpType::AllReduceProd:
        return infinicclProd;
    case OpType::AllReduceMin:
        return infinicclMin;
    case OpType::AllReduceMax:
        return infinicclMax;
    case OpType::AllReduceAvg:
        return infinicclAvg;
    default:
        IT_TODO_HALT_MSG("Unsupported InfiniCCL reduction");
    }
}

class AllReduceInfiniCcl final : public CommunicationKernel {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<AllReduceBaseObj>(_op);
        auto input = op->getInputs(0);
        auto output = op->getOutput();
        const auto &comm = getInfiniCcl(context);
        checkInfiniCcl(
            infinicclAllReduce(input->getRawDataPtr<void *>(),
                               output->getRawDataPtr<void *>(), input->size(),
                               toInfiniCclDataType(op->getDType()),
                               toInfiniCclReduction(op->getOpType()),
                               comm.getComm(), getInfiniStream(context)),
            "infinicclAllReduce");
    }
};

class AllGatherInfiniCcl final : public CommunicationKernel {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<AllGatherObj>(_op);
        const auto &comm = getInfiniCcl(context);
        IT_ASSERT(op->getWorldSize() == comm.getWorldSize(),
                  "AllGather world size does not match communicator");

        auto input = op->getInputs(0);
        const auto bytes = input->getBytes();
        auto gathered = const_cast<RuntimeObj *>(context)->allocBlob(
            bytes * static_cast<size_t>(comm.getWorldSize()));
        checkInfiniCcl(
            infinicclAllGather(input->getRawDataPtr<void *>(),
                               gathered->getPtr<void *>(), input->size(),
                               toInfiniCclDataType(input->getDType()),
                               comm.getComm(), getInfiniStream(context)),
            "infinicclAllGather");

        auto *base = static_cast<char *>(gathered->getPtr<void *>());
        for (int rank = 0; rank < comm.getWorldSize(); ++rank) {
            context->copyBlobInside(
                op->getOutput(rank)->getRawDataPtr<void *>(),
                base + rank * bytes, bytes);
        }
    }
};

class BroadcastInfiniCcl final : public CommunicationKernel {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<BroadcastObj>(_op);
        auto input = op->getInputs(0);
        auto output = op->getOutput();
        const auto &comm = getInfiniCcl(context);
        checkInfiniCcl(
            infinicclBroadcast(
                input->getRawDataPtr<void *>(), output->getRawDataPtr<void *>(),
                input->size(), toInfiniCclDataType(input->getDType()),
                op->getRoot(), comm.getComm(), getInfiniStream(context)),
            "infinicclBroadcast");
    }
};

class SendInfiniCcl final : public CommunicationKernel {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<SendObj>(_op);
        const auto &comm = getInfiniCcl(context);
        if (comm.getRank() != op->getSourceRank())
            return;
        auto input = op->getInputs(0);
        checkInfiniCcl(infinicclSend(input->getRawDataPtr<void *>(),
                                     input->size(),
                                     toInfiniCclDataType(input->getDType()),
                                     op->getDestinationRank(), comm.getComm(),
                                     getInfiniStream(context)),
                       "infinicclSend");
    }
};

class RecvInfiniCcl final : public CommunicationKernel {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<RecvObj>(_op);
        const auto &comm = getInfiniCcl(context);
        if (comm.getRank() != op->getDestinationRank())
            return;
        auto output = op->getOutput();
        checkInfiniCcl(infinicclRecv(output->getRawDataPtr<void *>(),
                                     output->size(),
                                     toInfiniCclDataType(op->getDType()),
                                     op->getSourceRank(), comm.getComm(),
                                     getInfiniStream(context)),
                       "infinicclRecv");
    }
};

REGISTER_KERNEL(ExecutionProvider::Infini, OpType::AllReduceSum,
                AllReduceInfiniCcl, "AllReduceSum_InfiniCCL");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::AllReduceProd,
                AllReduceInfiniCcl, "AllReduceProd_InfiniCCL");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::AllReduceMin,
                AllReduceInfiniCcl, "AllReduceMin_InfiniCCL");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::AllReduceMax,
                AllReduceInfiniCcl, "AllReduceMax_InfiniCCL");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::AllReduceAvg,
                AllReduceInfiniCcl, "AllReduceAvg_InfiniCCL");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::AllGather,
                AllGatherInfiniCcl, "AllGather_InfiniCCL");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Broadcast,
                BroadcastInfiniCcl, "Broadcast_InfiniCCL");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Send, SendInfiniCcl,
                "Send_InfiniCCL");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Recv, RecvInfiniCcl,
                "Recv_InfiniCCL");

} // namespace
} // namespace infini
#endif

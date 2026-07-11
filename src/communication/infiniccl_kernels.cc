#ifdef USE_INFINICCL
#include "communication/infiniccl_communicator.h"
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

    PerfRecord tune(const Operator &op, const RuntimeObj *context) const override {
        return make_ref<PerfRecordObj>(timeit([&]() { compute(op, context); }));
    }

    virtual void compute(const Operator &op, const RuntimeObj *context) const = 0;
};

const InfiniCclCommunicatorObj &getInfiniCcl(const RuntimeObj *context) {
    return dynamic_cast<const InfiniCclCommunicatorObj &>(
        context->getCommunicator());
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
    void compute(const Operator &_op, const RuntimeObj *context) const override {
        auto op = as<AllReduceBaseObj>(_op);
        auto input = op->getInputs(0);
        auto output = op->getOutput();
        const auto &comm = getInfiniCcl(context);
        checkInfiniCcl(
            infinicclAllReduce(input->getRawDataPtr<void *>(),
                               output->getRawDataPtr<void *>(), input->size(),
                               toInfiniCclDataType(op->getDType()),
                               toInfiniCclReduction(op->getOpType()),
                               comm.getComm(), nullptr),
            "infinicclAllReduce");
    }
};

class AllGatherInfiniCcl final : public CommunicationKernel {
    void compute(const Operator &_op, const RuntimeObj *context) const override {
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
                               comm.getComm(), nullptr),
            "infinicclAllGather");

        auto *base = static_cast<char *>(gathered->getPtr<void *>());
        for (int rank = 0; rank < comm.getWorldSize(); ++rank) {
            context->copyBlobInside(op->getOutput(rank)->getRawDataPtr<void *>(),
                                    base + rank * bytes, bytes);
        }
    }
};

class BroadcastInfiniCcl final : public CommunicationKernel {
    void compute(const Operator &_op, const RuntimeObj *context) const override {
        auto op = as<BroadcastObj>(_op);
        auto input = op->getInputs(0);
        auto output = op->getOutput();
        const auto &comm = getInfiniCcl(context);
        checkInfiniCcl(
            infinicclBroadcast(input->getRawDataPtr<void *>(),
                               output->getRawDataPtr<void *>(), input->size(),
                               toInfiniCclDataType(input->getDType()),
                               op->getRoot(), comm.getComm(), nullptr),
            "infinicclBroadcast");
    }
};

class SendInfiniCcl final : public CommunicationKernel {
    void compute(const Operator &_op, const RuntimeObj *context) const override {
        auto op = as<SendObj>(_op);
        const auto &comm = getInfiniCcl(context);
        if (comm.getRank() != op->getSourceRank()) return;
        auto input = op->getInputs(0);
        checkInfiniCcl(
            infinicclSend(input->getRawDataPtr<void *>(), input->size(),
                          toInfiniCclDataType(input->getDType()),
                          op->getDestinationRank(), comm.getComm(), nullptr),
            "infinicclSend");
    }
};

class RecvInfiniCcl final : public CommunicationKernel {
    void compute(const Operator &_op, const RuntimeObj *context) const override {
        auto op = as<RecvObj>(_op);
        const auto &comm = getInfiniCcl(context);
        if (comm.getRank() != op->getDestinationRank()) return;
        auto output = op->getOutput();
        checkInfiniCcl(
            infinicclRecv(output->getRawDataPtr<void *>(), output->size(),
                          toInfiniCclDataType(op->getDType()),
                          op->getSourceRank(), comm.getComm(), nullptr),
            "infinicclRecv");
    }
};

#define REGISTER_INFINICCL_KERNELS(DEVICE, SUFFIX)                             \
    REGISTER_KERNEL(DEVICE, OpType::AllReduceSum, AllReduceInfiniCcl,           \
                    "AllReduceSum_InfiniCCL_" SUFFIX);                         \
    REGISTER_KERNEL(DEVICE, OpType::AllReduceProd, AllReduceInfiniCcl,          \
                    "AllReduceProd_InfiniCCL_" SUFFIX);                        \
    REGISTER_KERNEL(DEVICE, OpType::AllReduceMin, AllReduceInfiniCcl,           \
                    "AllReduceMin_InfiniCCL_" SUFFIX);                         \
    REGISTER_KERNEL(DEVICE, OpType::AllReduceMax, AllReduceInfiniCcl,           \
                    "AllReduceMax_InfiniCCL_" SUFFIX);                         \
    REGISTER_KERNEL(DEVICE, OpType::AllReduceAvg, AllReduceInfiniCcl,           \
                    "AllReduceAvg_InfiniCCL_" SUFFIX);                         \
    REGISTER_KERNEL(DEVICE, OpType::AllGather, AllGatherInfiniCcl,              \
                    "AllGather_InfiniCCL_" SUFFIX);                            \
    REGISTER_KERNEL(DEVICE, OpType::Broadcast, BroadcastInfiniCcl,              \
                    "Broadcast_InfiniCCL_" SUFFIX);                            \
    REGISTER_KERNEL(DEVICE, OpType::Send, SendInfiniCcl,                         \
                    "Send_InfiniCCL_" SUFFIX);                                 \
    REGISTER_KERNEL(DEVICE, OpType::Recv, RecvInfiniCcl,                         \
                    "Recv_InfiniCCL_" SUFFIX)

#ifdef USE_CUDA
REGISTER_INFINICCL_KERNELS(Device::CUDA, "CUDA");
#endif
#ifdef USE_BANG
REGISTER_INFINICCL_KERNELS(Device::BANG, "BANG");
#endif
#ifdef USE_ILUVATAR
REGISTER_INFINICCL_KERNELS(Device::ILUVATAR, "ILUVATAR");
#endif
#ifdef USE_METAX
REGISTER_INFINICCL_KERNELS(Device::METAX, "METAX");
#endif
#ifdef USE_MOORE
REGISTER_INFINICCL_KERNELS(Device::MOORE, "MOORE");
#endif

#undef REGISTER_INFINICCL_KERNELS

} // namespace
} // namespace infini
#endif

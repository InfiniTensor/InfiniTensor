#include "operators/pooling.h"
#include "core/kernel.h"
#include "utils/infiniop_utils.h"

namespace infini {

class PoolingOp : public Kernel {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<PoolingObj>(_op);
        void *const xData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const yData = (op->getOutput()->getRawDataPtr<void *>());

        if (op->getOpType() == OpType::MaxPool) {
            // get workspace
            uint64_t workspace_size = 0;
            CHECK_ERROR(infiniopGetMaxPoolWorkspaceSize(
                (infiniopMaxPoolDescriptor_t)op->getOpDesc(), &workspace_size));
            IT_ASSERT(workspace_size <= context->getWorkspaceSize());
            void *workspace = context->getWorkspace(workspace_size);

            // execute op
            CHECK_ERROR(infiniopMaxPool(
                (infiniopMaxPoolDescriptor_t)op->getOpDesc(), workspace,
                workspace_size, yData, xData, context->getCurrentStream()));
        } else if (op->getOpType() == OpType::AveragePool) {
            // get workspace
            uint64_t workspace_size = 0;
            CHECK_ERROR(infiniopGetAvgPoolWorkspaceSize(
                (infiniopAvgPoolDescriptor_t)op->getOpDesc(), &workspace_size));
            IT_ASSERT(workspace_size <= context->getWorkspaceSize());
            void *workspace = context->getWorkspace(workspace_size);

            // execute op
            CHECK_ERROR(infiniopAvgPool(
                (infiniopAvgPoolDescriptor_t)op->getOpDesc(), workspace,
                workspace_size, yData, xData, context->getCurrentStream()));
        } else {
            IT_TODO_HALT();
        }
    }

    PerfRecord tune(const Operator &_op,
                    const RuntimeObj *context) const override {
        // TODO: tune should be in infiniop
        return PerfRecord();
    }

    void compute(const Operator &op, const PerfRecord &record,
                 const RuntimeObj *context) const override {
        compute(op, context);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::MaxPool, PoolingOp,
                "Pooling_infiniop_cuda");
REGISTER_KERNEL(Device::CPU, OpType::MaxPool, PoolingOp,
                "Pooling_infiniop_cpu");
REGISTER_KERNEL(Device::CUDA, OpType::AveragePool, PoolingOp,
                "Pooling_infiniop_cuda");
REGISTER_KERNEL(Device::CPU, OpType::AveragePool, PoolingOp,
                "Pooling_infiniop_cpu");
}; // namespace infini

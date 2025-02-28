#include "operators/reduce.h"
#include "core/kernel.h"
#include "utils/infiniop_utils.h"

namespace infini {

class ReduceOp : public Kernel {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<ReduceBaseObj>(_op);
        void *const xData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const yData = (op->getOutput()->getRawDataPtr<void *>());

        if (op->getOpType() == OpType::ReduceMin) {
            // execute op
            Shape xShape = op->getInputs(0)->getDims();
            Shape yShape = op->getOutput()->getDims();
            uint64_t workspace_size = 0;
            CHECK_ERROR(infiniopGetReduceMinWorkspaceSize(
                (infiniopReduceMinDescriptor_t)op->getOpDesc(),
                &workspace_size));
            IT_ASSERT(workspace_size <= context->getWorkspaceSize());
            void *workspace = context->getWorkspace(workspace_size);
            CHECK_ERROR(infiniopReduceMin(
                (infiniopReduceMinDescriptor_t)op->getOpDesc(), workspace,
                workspace_size, yData, xData, context->getCurrentStream()));
        } else if (op->getOpType() == OpType::ReduceMax) {
            // execute op
            uint64_t workspace_size = 0;
            CHECK_ERROR(infiniopGetReduceMaxWorkspaceSize(
                (infiniopReduceMaxDescriptor_t)op->getOpDesc(),
                &workspace_size));
            IT_ASSERT(workspace_size <= context->getWorkspaceSize());
            void *workspace = context->getWorkspace(workspace_size);
            CHECK_ERROR(infiniopReduceMax(
                (infiniopReduceMaxDescriptor_t)op->getOpDesc(), workspace,
                workspace_size, yData, xData, context->getCurrentStream()));
        } else if (op->getOpType() == OpType::ReduceMean) {
            // execute op
            uint64_t workspace_size = 0;
            CHECK_ERROR(infiniopGetReduceMeanWorkspaceSize(
                (infiniopReduceMeanDescriptor_t)op->getOpDesc(),
                &workspace_size));
            IT_ASSERT(workspace_size <= context->getWorkspaceSize());
            void *workspace = context->getWorkspace(workspace_size);
            CHECK_ERROR(infiniopReduceMean(
                (infiniopReduceMeanDescriptor_t)op->getOpDesc(), workspace,
                workspace_size, yData, xData, context->getCurrentStream()));
        } else {
            IT_TODO_HALT();
        }
    }

    PerfRecord tune(const Operator &_op,
                    const RuntimeObj *_context) const override {
        // TODO: tune should be in infiniop
        return PerfRecord();
    }

    void compute(const Operator &op, const PerfRecord &record,
                 const RuntimeObj *context) const override {
        compute(op, context);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::ReduceMin, ReduceOp,
                "ReduceMin_infiniop_CUDA");
REGISTER_KERNEL(Device::CPU, OpType::ReduceMin, ReduceOp,
                "ReduceMin_infiniop_CPU");
REGISTER_KERNEL(Device::CUDA, OpType::ReduceMax, ReduceOp,
                "ReduceMax_infiniop_CUDA");
REGISTER_KERNEL(Device::CPU, OpType::ReduceMax, ReduceOp,
                "ReduceMax_infiniop_CPU");
REGISTER_KERNEL(Device::CUDA, OpType::ReduceMean, ReduceOp,
                "ReduceMean_infiniop_CUDA");
REGISTER_KERNEL(Device::CPU, OpType::ReduceMean, ReduceOp,
                "ReduceMean_infiniop_CPU");
}; // namespace infini

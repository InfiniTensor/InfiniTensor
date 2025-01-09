#include "operators/global_pool.h"
#include "core/kernel.h"
#include "utils/infiniop_utils.h"

namespace infini {

class GlobalPoolOp : public Kernel {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<GlobalPoolObj>(_op);
        void *const xData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const yData = (op->getOutput()->getRawDataPtr<void *>());

        if (op->getOpType() == OpType::GlobalAveragePool) {
            // get workspace size
            uint64_t workspace_size = 0;
            CHECK_ERROR(infiniopGetGlobalAvgPoolWorkspaceSize(
                (infiniopGlobalAvgPoolDescriptor_t)op->getOpDesc(),
                &workspace_size));
            IT_ASSERT(workspace_size <= context->getWorkspaceSize());
            void *workspace = context->getWorkspace(workspace_size);

            // execute op
            CHECK_ERROR(infiniopGlobalAvgPool(
                (infiniopGlobalAvgPoolDescriptor_t)op->getOpDesc(), workspace,
                workspace_size, yData, xData, context->getCurrentStream()));
        } else {
            IT_TODO_HALT();
        }
    }

    PerfRecord tune(const Operator &op,
                    const RuntimeObj *context) const override {
        // TODO: tune should be in infiniop
        return PerfRecord();
    }

    void compute(const Operator &op, const PerfRecord &record,
                 const RuntimeObj *context) const override {
        compute(op, context);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::GlobalAveragePool, GlobalPoolOp,
                "GlobalAvgPool_infiniop_CUDA");
REGISTER_KERNEL(Device::CPU, OpType::GlobalAveragePool, GlobalPoolOp,
                "GlobalAvgPool_infiniop_CPU");
}; // namespace infini

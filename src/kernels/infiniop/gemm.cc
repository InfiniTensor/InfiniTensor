#include "operators/gemm.h"
#include "core/kernel.h"
#include "utils/infiniop_utils.h"

namespace infini {

class GemmOp : public Kernel {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<GemmObj>(_op);
        void *const yData = (op->getOutput()->getRawDataPtr<void *>());
        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = op->numInputs() == 3
                                ? (op->getInputs(2)->getRawDataPtr<void *>())
                                : nullptr;

        // get workspace size and allocate workspace
        uint64_t workspace_size = 0;
        CHECK_ERROR(infiniopGetGEMMWorkspaceSize(
            (infiniopGEMMDescriptor_t)op->getOpDesc(), &workspace_size));
        IT_ASSERT(workspace_size <= context->getWorkspaceSize());
        void *workspace = context->getWorkspace(workspace_size);
        CHECK_ERROR(infiniopGEMM((infiniopGEMMDescriptor_t)op->getOpDesc(),
                                 workspace, workspace_size, yData, aData, bData,
                                 cData, context->getCurrentStream()));
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

REGISTER_KERNEL(Device::CUDA, OpType::Gemm, GemmOp, "Gemm_infiniop_CUDA");
REGISTER_KERNEL(Device::CPU, OpType::Gemm, GemmOp, "Gemm_infiniop_CPU");
}; // namespace infini

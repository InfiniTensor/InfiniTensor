#include "core/kernel.h"
#include "operators/element_wise.h"

namespace infini {
class ElementWiseOp : public Kernel {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<ElementWiseObj>(_op);
        op->createOpDesc();
        auto type = op->getOpType();
        void *yData = (op->getOutput()->getRawDataPtr<void *>());
        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        size_t workspace_size = 0;
        if (type == OpType::Add) {
            CHECK_INFINI_ERROR(infiniopGetAddWorkspaceSize(
                (infiniopAddDescriptor_t)(op->getInfiniOpDesc()),
                &workspace_size));
            void *workspace = context->getWorkspace(workspace_size);
            CHECK_INFINI_ERROR(
                infiniopAdd((infiniopAddDescriptor_t)(op->getInfiniOpDesc()),
                            workspace, workspace_size, yData, aData, bData,
                            context->getCurrentStream()));
        } else if (type == OpType::Mul) {
            CHECK_INFINI_ERROR(infiniopGetMulWorkspaceSize(
                (infiniopMulDescriptor_t)op->getInfiniOpDesc(),
                &workspace_size));
            void *workspace = context->getWorkspace(workspace_size);
            CHECK_INFINI_ERROR(
                infiniopMul((infiniopMulDescriptor_t)op->getInfiniOpDesc(),
                            workspace, workspace_size, yData, aData, bData,
                            context->getCurrentStream()));
        } else if (type == OpType::Sub) {
            CHECK_INFINI_ERROR(infiniopGetSubWorkspaceSize(
                (infiniopSubDescriptor_t)op->getInfiniOpDesc(),
                &workspace_size));
            void *workspace = context->getWorkspace(workspace_size);
            CHECK_INFINI_ERROR(
                infiniopSub((infiniopSubDescriptor_t)op->getInfiniOpDesc(),
                            workspace, workspace_size, yData, aData, bData,
                            context->getCurrentStream()));
        } else {
            IT_TODO_HALT_MSG("ElemenWise operator not supported");
        }
    }

    PerfRecord tune(const Operator &op,
                    const RuntimeObj *context) const override {
        // only for virtual function call
        return PerfRecord();
    }

    void compute(const Operator &op, const PerfRecord &record,
                 const RuntimeObj *context) const override {
        compute(op, context);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Add, ElementWiseOp, "Add_infiniop_CUDA");
REGISTER_KERNEL(Device::CPU, OpType::Add, ElementWiseOp, "Add_infiniop_CPU");
REGISTER_KERNEL(Device::BANG, OpType::Add, ElementWiseOp, "Add_infiniop_BANG");
REGISTER_KERNEL(Device::ASCEND, OpType::Add, ElementWiseOp,
                "Add_infiniop_ASCEND");
REGISTER_KERNEL(Device::KUNLUN, OpType::Add, ElementWiseOp,
                "Add_infiniop_KUNLUN");
REGISTER_KERNEL(Device::METAX, OpType::Add, ElementWiseOp,
                "Add_infiniop_METAX");
REGISTER_KERNEL(Device::MOORE, OpType::Add, ElementWiseOp,
                "Add_infiniop_MOORE");
REGISTER_KERNEL(Device::ILUVATAR, OpType::Add, ElementWiseOp,
                "Add_infiniop_ILUVATAR");
} // namespace infini

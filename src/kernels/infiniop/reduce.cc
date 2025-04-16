#include "operators/reduce.h"
#include "infini_operators.h"
#include "core/kernel.h"
#include "utils/infiniop_utils.h"

namespace infini {

class ReduceOp : public Kernel {
    void compute(const Operator &_op, const RuntimeObj *context) const override {
        auto op = as<ReduceBaseObj>(_op);

        auto input = op->getInputs(0)->getRawDataPtr<void *>();
        auto output = op->getOutput()->getRawDataPtr<void *>();

        // 获取工作区大小
        uint64_t workspace_size = 0;
        

       // 根据 ReduceType 执行
    auto opType = op->getOpType();
    // int op_value=0;

    if (opType == OpType::ReduceMax) {
        // op_value=0;
        CHECK_ERROR(infiniopGetReduceMaxWorkspaceSize(
            (infiniopReduceMaxDescriptor_t)op->getOpDesc(), &workspace_size));

        // 分配临时 workspace（栈分配 or 你可以改成 allocator 分配）
        IT_ASSERT(workspace_size <= context->getWorkspaceSize());
        void *workspace = context->getWorkspace(workspace_size);
        CHECK_ERROR(infiniopReduceMax(
            (infiniopReduceMaxDescriptor_t)op->getOpDesc(),
            workspace,
            workspace_size,
            output,
            input,
            context->getCurrentStream()));
    } else if (opType == OpType::ReduceMean) {
        // op_value=1;
        CHECK_ERROR(infiniopGetReduceMeanWorkspaceSize(
            (infiniopReduceMeanDescriptor_t)op->getOpDesc(), &workspace_size));

        // 分配临时 workspace（栈分配 or 你可以改成 allocator 分配）
        IT_ASSERT(workspace_size <= context->getWorkspaceSize());
        void *workspace = context->getWorkspace(workspace_size);
        CHECK_ERROR(infiniopReduceMean(
            (infiniopReduceMeanDescriptor_t)op->getOpDesc(),
            workspace,
            workspace_size,
            output,
            input,
            context->getCurrentStream()));
    } else if (opType == OpType::ReduceMin) {
        // op_value=2;
        CHECK_ERROR(infiniopGetReduceMinWorkspaceSize(
            (infiniopReduceMinDescriptor_t)op->getOpDesc(), &workspace_size));

        // 分配临时 workspace（栈分配 or 你可以改成 allocator 分配）
        IT_ASSERT(workspace_size <= context->getWorkspaceSize());
        void *workspace = context->getWorkspace(workspace_size);
        CHECK_ERROR(infiniopReduceMin(
            (infiniopReduceMinDescriptor_t)op->getOpDesc(),
            workspace,
            workspace_size,
            output,
            input,
            context->getCurrentStream()));
    } else {
        IT_TODO_HALT();  // 如果有未支持的类型
    }
}


    PerfRecord tune(const Operator &_op,
                    const RuntimeObj *_context) const override {
        return PerfRecord();  // 暂不调优
    }

    void compute(const Operator &_op,
                 const PerfRecord &record,
                 const RuntimeObj *context) const override {
        compute(_op, context);  // 复用 compute 逻辑
    }

};

// 注册 CPU 内核
// REGISTER_KERNEL(Device::CPU, OpType::ReduceSum, ReduceOp, "ReduceSum_infiniop_CPU");
REGISTER_KERNEL(Device::CPU, OpType::ReduceMax, ReduceOp, "ReduceMax_infiniop_CPU");
REGISTER_KERNEL(Device::CPU, OpType::ReduceMin, ReduceOp, "ReduceMin_infiniop_CPU");
REGISTER_KERNEL(Device::CPU, OpType::ReduceMean, ReduceOp, "ReduceMean_infiniop_CPU");

} // namespace infini

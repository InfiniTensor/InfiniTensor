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
        auto x_dim = op->getInputs(0)->getDims();
        auto y_dim = op->getOutput()->getDims();
        auto dType = toInfiniopDataLayout(_op->getDType().getIndex());

        if (op->getOpType() == OpType::GlobalAveragePool) {
            auto x_shape = toInfiniopShape(x_dim);
            auto y_shape = toInfiniopShape(y_dim);
            // create tensor descriptor
            infiniopTensorDescriptor_t x_tensor = new TensorDescriptor;
            CHECK_ERROR(infiniopCreateTensorDescriptor(
                &x_tensor, x_dim.size(), x_shape.data(), nullptr, dType));
            infiniopTensorDescriptor_t y_tensor = new TensorDescriptor;
            CHECK_ERROR(infiniopCreateTensorDescriptor(
                &y_tensor, y_dim.size(), y_shape.data(), nullptr, dType));
            // create op descriptor
            infiniopGlobalAvgPoolDescriptor_t op_desc =
                new GlobalAvgPoolDescriptor;
            CHECK_ERROR(infiniopCreateGlobalAvgPoolDescriptor(
                context->opHandle(), &op_desc, y_tensor, x_tensor));
            uint64_t workspace_size = 0;
            CHECK_ERROR(infiniopGetGlobalAvgPoolWorkspaceSize(op_desc,
                                                              &workspace_size));
            if (workspace_size > context->getWorkspaceSize()) {
                IT_TODO_HALT();
            }
            void *workspace = context->getWorkspace(workspace_size);
            // execute op (TODO: 前面创建 op_desc 的步骤应当挪到计算函数外）
            CHECK_ERROR(infiniopGlobalAvgPool(
                op_desc, workspace, workspace_size, yData, xData, nullptr));

            // 销毁
            infiniopDestroyTensorDescriptor(x_tensor);
            infiniopDestroyTensorDescriptor(y_tensor);
            infiniopDestroyGlobalAvgPoolDescriptor(op_desc);
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
}; // namespace infini

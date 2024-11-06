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
        auto x_dim = op->getInputs(0)->getDims();
        auto y_dim = op->getOutput()->getDims();
        const auto [n, c, h, w, kh, kw] = op->getNCHWRS();
        const auto [ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();

        uint64_t kernel_shape[2] = {(uint64_t)kh, (uint64_t)kw};
        uint64_t pads[2] = {(uint64_t)ph, (uint64_t)pw};
        int64_t strides[2] = {(int64_t)sh, (int64_t)sw};

        auto dType = toInfiniopDataLayout(_op->getDType().getIndex());

        auto x_shape = toInfiniopShape(x_dim);
        auto y_shape = toInfiniopShape(y_dim);
        // create tensor descriptor
        infiniopTensorDescriptor_t x_tensor;
        CHECK_ERROR(infiniopCreateTensorDescriptor(
            &x_tensor, x_dim.size(), x_shape.data(), nullptr, dType));
        infiniopTensorDescriptor_t y_tensor;
        CHECK_ERROR(infiniopCreateTensorDescriptor(
            &y_tensor, y_dim.size(), y_shape.data(), nullptr, dType));

        if (op->getOpType() == OpType::MaxPool) {
            // create op descriptor
            infiniopMaxPoolDescriptor_t op_desc;
            CHECK_ERROR(infiniopCreateMaxPoolDescriptor(
                context->opHandle(), &op_desc, y_tensor, x_tensor, kernel_shape,
                pads, strides, 2));
            uint64_t workspace_size = 0;
            CHECK_ERROR(
                infiniopGetMaxPoolWorkspaceSize(op_desc, &workspace_size));
            IT_ASSERT(workspace_size <= context->getWorkspaceSize());
            void *workspace = context->getWorkspace(workspace_size);
            // execute op (TODO: 前面创建 op_desc 的步骤应当挪到计算函数外）
            CHECK_ERROR(infiniopMaxPool(op_desc, workspace, workspace_size,
                                        yData, xData, nullptr));
            CHECK_ERROR(infiniopDestroyMaxPoolDescriptor(op_desc));
        } else if (op->getOpType() == OpType::AveragePool) {
            // create op descriptor
            infiniopAvgPoolDescriptor_t op_desc;
            CHECK_ERROR(infiniopCreateAvgPoolDescriptor(
                context->opHandle(), &op_desc, y_tensor, x_tensor, kernel_shape,
                pads, strides, 2));
            uint64_t workspace_size = 0;
            CHECK_ERROR(
                infiniopGetAvgPoolWorkspaceSize(op_desc, &workspace_size));
            IT_ASSERT(workspace_size <= context->getWorkspaceSize());
            void *workspace = context->getWorkspace(workspace_size);
            // execute op (TODO: 前面创建 op_desc 的步骤应当挪到计算函数外）
            CHECK_ERROR(infiniopAvgPool(op_desc, workspace, workspace_size,
                                        yData, xData, nullptr));
            CHECK_ERROR(infiniopDestroyAvgPoolDescriptor(op_desc));
        } else {
            IT_TODO_HALT();
        }

        // destroy tensor descriptor
        CHECK_ERROR(infiniopDestroyTensorDescriptor(y_tensor));
        CHECK_ERROR(infiniopDestroyTensorDescriptor(x_tensor));
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

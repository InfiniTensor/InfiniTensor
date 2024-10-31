#include "operators/conv.h"
#include "core/kernel.h"
#include "utils/infiniop_utils.h"

namespace infini {

class ConvOp : public Kernel {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<ConvObj>(_op);
        void *const xData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const wData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const yData = (op->getOutput()->getRawDataPtr<void *>());
        auto x_dim = op->getInputs(0)->getDims();
        auto w_dim = op->getInputs(1)->getDims();
        auto y_dim = op->getOutput()->getDims();
        const auto [ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
        uint64_t pads[2] = {(uint64_t)ph, (uint64_t)pw};
        uint64_t strides[2] = {(uint64_t)sh, (uint64_t)sw};
        uint64_t dilations[2] = {(uint64_t)dh, (uint64_t)dw};

        auto dType = toInfiniopDataLayout(_op->getDType().getIndex());

        auto x_shape = toInfiniopShape(x_dim);
        auto w_shape = toInfiniopShape(w_dim);
        auto y_shape = toInfiniopShape(y_dim);
        // create tensor descriptor
        infiniopTensorDescriptor_t x_tensor = new TensorDescriptor;
        CHECK_ERROR(infiniopCreateTensorDescriptor(
            &x_tensor, x_dim.size(), x_shape.data(), nullptr, dType));
        infiniopTensorDescriptor_t w_tensor = new TensorDescriptor;
        CHECK_ERROR(infiniopCreateTensorDescriptor(
            &w_tensor, w_dim.size(), w_shape.data(), nullptr, dType));
        infiniopTensorDescriptor_t y_tensor = new TensorDescriptor;
        CHECK_ERROR(infiniopCreateTensorDescriptor(
            &y_tensor, y_dim.size(), y_shape.data(), nullptr, dType));
        // create op descriptor
        infiniopConvDescriptor_t op_desc = new ConvDescriptor;
        CHECK_ERROR(infiniopCreateConvDescriptor(context->opHandle(), &op_desc,
                                                 y_tensor, x_tensor, w_tensor,
                                                 pads, strides, dilations, 2));
        uint64_t workspace_size = 0;
        CHECK_ERROR(infiniopGetConvWorkspaceSize(op_desc, &workspace_size));
        if (workspace_size > context->getWorkspaceSize()) {
            IT_TODO_HALT();
        }
        void *workspace = context->getWorkspace(workspace_size);
        // execute op (TODO: 前面创建 op_desc 的步骤应当挪到计算函数外）
        CHECK_ERROR(infiniopConv(op_desc, workspace, workspace_size, yData,
                                 xData, wData, nullptr));

        // 销毁
        CHECK_ERROR(infiniopDestroyTensorDescriptor(y_tensor));
        CHECK_ERROR(infiniopDestroyTensorDescriptor(x_tensor));
        CHECK_ERROR(infiniopDestroyTensorDescriptor(w_tensor));
        CHECK_ERROR(infiniopDestroyConvDescriptor(op_desc));
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

REGISTER_KERNEL(Device::CUDA, OpType::Conv, ConvOp, "Conv_infiniop_cuda");
REGISTER_KERNEL(Device::CPU, OpType::Conv, ConvOp, "Conv_infiniop_cpu");
}; // namespace infini

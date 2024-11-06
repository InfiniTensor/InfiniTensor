#include "operators/unary.h"
#include "core/kernel.h"
#include "utils/infiniop_utils.h"

namespace infini {

class UnaryOp : public Kernel {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        void *const xData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const yData = (op->getOutput()->getRawDataPtr<void *>());
        auto x_dim = op->getInputs(0)->getDims();
        auto y_dim = op->getOutput()->getDims();
        auto dType = toInfiniopDataLayout(_op->getDType().getIndex());

        if (op->getOpType() == OpType::Relu) {
            auto x_shape = toInfiniopShape(x_dim);
            auto y_shape = toInfiniopShape(y_dim);
            // create tensor descriptor
            infiniopTensorDescriptor_t x_tensor;
            CHECK_ERROR(infiniopCreateTensorDescriptor(
                &x_tensor, x_dim.size(), x_shape.data(), nullptr, dType));
            infiniopTensorDescriptor_t y_tensor;
            CHECK_ERROR(infiniopCreateTensorDescriptor(
                &y_tensor, y_dim.size(), y_shape.data(), nullptr, dType));
            // create op descriptor
            infiniopReluDescriptor_t op_desc;
            CHECK_ERROR(infiniopCreateReluDescriptor(
                _context->opHandle(), &op_desc, y_tensor, x_tensor));
            // execute op (TODO: 前面创建 op_desc 的步骤应当挪到计算函数外）
            CHECK_ERROR(infiniopRelu(op_desc, yData, xData, nullptr));

            // 销毁
            CHECK_ERROR(infiniopDestroyTensorDescriptor(x_tensor));
            CHECK_ERROR(infiniopDestroyTensorDescriptor(y_tensor));
            CHECK_ERROR(infiniopDestroyReluDescriptor(op_desc));
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

REGISTER_KERNEL(Device::CUDA, OpType::Relu, UnaryOp, "Relu_infiniop_CUDA");
REGISTER_KERNEL(Device::CPU, OpType::Relu, UnaryOp, "Relu_infiniop_CPU");
}; // namespace infini

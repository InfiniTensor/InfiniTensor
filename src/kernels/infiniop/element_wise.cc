#include "operators/element_wise.h"
#include "core/kernel.h"
#include "utils/infiniop_utils.h"

namespace infini {

class ElementWiseOp : public Kernel {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto a_dim = op->getInputs(0)->getDims();
        auto b_dim = op->getInputs(1)->getDims();
        auto c_dim = op->getOutput()->getDims();
        auto dType = toInfiniopDataLayout(_op->getDType().getIndex());

        if (op->getOpType() == OpType::Add) {
            auto a_shape = toInfiniopShape(a_dim);
            auto b_shape = toInfiniopShape(b_dim);
            auto c_shape = toInfiniopShape(c_dim);
            // create tensor descriptor
            infiniopTensorDescriptor_t a_tensor;
            CHECK_ERROR(infiniopCreateTensorDescriptor(
                &a_tensor, a_dim.size(), a_shape.data(), nullptr, dType));
            infiniopTensorDescriptor_t b_tensor;
            CHECK_ERROR(infiniopCreateTensorDescriptor(
                &b_tensor, b_dim.size(), b_shape.data(), nullptr, dType));
            infiniopTensorDescriptor_t c_tensor;
            CHECK_ERROR(infiniopCreateTensorDescriptor(
                &c_tensor, c_dim.size(), c_shape.data(), nullptr, dType));
            // create op descriptor
            infiniopAddDescriptor_t op_desc;
            CHECK_ERROR(infiniopCreateAddDescriptor(
                _context->opHandle(), &op_desc, c_tensor, a_tensor, b_tensor));
            // execute op (TODO: 前面创建 op_desc 的步骤应当挪到计算函数外）
            CHECK_ERROR(infiniopAdd(op_desc, cData, aData, bData, nullptr));

            // 销毁
            infiniopDestroyTensorDescriptor(a_tensor);
            infiniopDestroyTensorDescriptor(b_tensor);
            infiniopDestroyTensorDescriptor(c_tensor);
            infiniopDestroyAddDescriptor(op_desc);
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

REGISTER_KERNEL(Device::CUDA, OpType::Add, ElementWiseOp, "Add_infiniop_cuda");
REGISTER_KERNEL(Device::CPU, OpType::Add, ElementWiseOp, "Add_infiniop_cpu");
}; // namespace infini

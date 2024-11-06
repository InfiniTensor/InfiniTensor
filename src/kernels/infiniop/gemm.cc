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
        auto a_dim = op->getInputs(0)->getDims();
        auto b_dim = op->getInputs(1)->getDims();
        auto c_dim =
            op->numInputs() == 3 ? op->getInputs(2)->getDims() : Shape{};
        auto y_dim = op->getOutput()->getDims();
        auto dType = toInfiniopDataLayout(_op->getDType().getIndex());

        // convert dim data to infiniop format
        auto y_shape = toInfiniopShape(y_dim);
        auto a_shape = toInfiniopShape(a_dim);
        auto b_shape = toInfiniopShape(b_dim);
        auto c_shape = toInfiniopShape(c_dim);
        // create tensor descriptor
        infiniopTensorDescriptor_t y_tensor;
        CHECK_ERROR(infiniopCreateTensorDescriptor(
            &y_tensor, y_dim.size(), y_shape.data(), nullptr, dType));
        infiniopTensorDescriptor_t a_tensor;
        CHECK_ERROR(infiniopCreateTensorDescriptor(
            &a_tensor, a_dim.size(), a_shape.data(), nullptr, dType));
        infiniopTensorDescriptor_t b_tensor;
        CHECK_ERROR(infiniopCreateTensorDescriptor(
            &b_tensor, b_dim.size(), b_shape.data(), nullptr, dType));
        infiniopTensorDescriptor_t c_tensor;
        if (cData != nullptr) {
            CHECK_ERROR(infiniopCreateTensorDescriptor(
                &c_tensor, c_dim.size(), c_shape.data(), nullptr, dType));
        }
        // create op descriptor
        infiniopGEMMDescriptor_t op_desc;
        CHECK_ERROR(infiniopCreateGEMMDescriptor(
            context->opHandle(), &op_desc, y_tensor, a_tensor, b_tensor,
            c_tensor, op->getAlpha(), op->getBeta(), op->getTransA(),
            op->getTransB()));
        // get workspace size and allocate workspace
        uint64_t workspace_size = 0;
        CHECK_ERROR(infiniopGetGEMMWorkspaceSize(op_desc, &workspace_size));
        IT_ASSERT(workspace_size <= context->getWorkspaceSize());
        void *workspace = context->getWorkspace(workspace_size);
        // execute op (TODO: 前面创建 op_desc 的步骤应当挪到计算函数外）
        CHECK_ERROR(infiniopGEMM(op_desc, workspace, workspace_size, yData,
                                 aData, bData, cData, nullptr));

        // 销毁
        CHECK_ERROR(infiniopDestroyTensorDescriptor(y_tensor));
        CHECK_ERROR(infiniopDestroyTensorDescriptor(a_tensor));
        CHECK_ERROR(infiniopDestroyTensorDescriptor(b_tensor));
        CHECK_ERROR(infiniopDestroyTensorDescriptor(c_tensor));
        CHECK_ERROR(infiniopDestroyGEMMDescriptor(op_desc));
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

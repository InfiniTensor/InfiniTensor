#include "operators/gemm.h"
#include "core/kernel.h"

namespace infini {

class GemmOp : public Kernel {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<GemmObj>(_op);
        std::cout << "===============================3.2.1" << std::endl;
        op->createOpDesc();
        void *yData = (op->getOutput()->getRawDataPtr<void *>());
        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        std::cout << "===============================3.2.2" << std::endl;
        size_t workspace_size = 0;
        CHECK_INFINI_ERROR(infiniopGetGemmWorkspaceSize(
            (infiniopGemmDescriptor_t)op->getInfiniOpDesc(), &workspace_size));
        std::cout << "===============================3.2.3" << std::endl;
        void *workspace = context->getWorkspace(workspace_size);
        std::cout << "===============================3.2.4" << std::endl;
        CHECK_INFINI_ERROR(infiniopGemm(
            (infiniopGemmDescriptor_t)op->getInfiniOpDesc(), workspace,
            workspace_size, yData, aData, bData, op->getAlpha(), op->getBeta(),
            context->getCurrentStream()));
        std::cout << "===============================3.2.5" << std::endl;
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

REGISTER_KERNEL(Device::CUDA, OpType::Gemm, GemmOp, "Gemm_infiniop_CUDA");
REGISTER_KERNEL(Device::CPU, OpType::Gemm, GemmOp, "Gemm_infiniop_CPU");
REGISTER_KERNEL(Device::BANG, OpType::Gemm, GemmOp, "Gemm_infiniop_BANG");
REGISTER_KERNEL(Device::ASCEND, OpType::Gemm, GemmOp, "Gemm_infiniop_ASCEND");
REGISTER_KERNEL(Device::KUNLUN, OpType::Gemm, GemmOp, "Gemm_infiniop_KUNLUN");
REGISTER_KERNEL(Device::METAX, OpType::Gemm, GemmOp, "Gemm_infiniop_METAX");
REGISTER_KERNEL(Device::MOORE, OpType::Gemm, GemmOp, "Gemm_infiniop_MOORE");
REGISTER_KERNEL(Device::ILUVATAR, OpType::Gemm, GemmOp,
                "Gemm_infiniop_ILUVATAR");
} // namespace infini
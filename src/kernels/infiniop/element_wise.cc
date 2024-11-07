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

        if (op->getOpType() == OpType::Add) {
            // execute op (TODO: 前面创建 op_desc 的步骤应当挪到计算函数外）
            CHECK_ERROR(infiniopAdd((infiniopAddDescriptor_t)op->getOpDesc(),
                                    cData, aData, bData, nullptr));
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

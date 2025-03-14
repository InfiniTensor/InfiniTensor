#include "operators/where.h"
#include "core/kernel.h"
#include "utils/infiniop_utils.h"

namespace infini {

class WhereOp : public Kernel {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<WhereObj>(_op);
        void *const xData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const yData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const conditions = (op->getInputs(2)->getRawDataPtr<void *>());
        void *const dst = (op->getOutput()->getRawDataPtr<void *>());
        
        if (op->getOpType() == OpType::Where) {
            CHECK_ERROR(infiniopWhere(
                (infiniopWhereDescriptor_t)op->getOpDesc(),
                dst, xData, yData, conditions, context->getCurrentStream()));
        } else {
            IT_TODO_HALT();
        }
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

REGISTER_KERNEL(Device::CUDA, OpType::Where, WhereOp,
                "Where_infiniop_cuda");
REGISTER_KERNEL(Device::CPU, OpType::Where, WhereOp,
                "Where_infiniop_cpu");
}; // namespace infini

#include "operators/where.h"
#include "infini_operators.h"
#include "core/kernel.h"
#include "utils/infiniop_utils.h"
namespace infini {
class WhereOp : public Kernel {
    void compute(const Operator &_op,
                    const RuntimeObj *context) const override {
        auto op = as<WhereObj>(_op);

        auto cond = op->getInputs(2)->getRawDataPtr<void*>();
        auto x = op->getInputs(0)->getRawDataPtr<void*>();
        auto y = op->getInputs(1)->getRawDataPtr<void*>();
        auto out = op->getOutput()->getRawDataPtr<void*>();
        

        if (op->getOpType() == OpType::Where) {
            CHECK_ERROR(infiniopWhere(
                (infiniopWhereDescriptor_t)op->getOpDesc(),
                 cond, x, y,out, context->getCurrentStream()));
        } else {
            IT_TODO_HALT();
        }
    }

    PerfRecord tune(const Operator &_op,
                    const RuntimeObj *_context) const override {
        return PerfRecord();
    }

    void compute(const Operator &_op,
                    const PerfRecord &record,
                    const RuntimeObj *context) const override {
        compute(_op, context);
    }
};
    
REGISTER_KERNEL(Device::CPU,OpType::Where,WhereOp,"Where_infiniop_CPU");
}
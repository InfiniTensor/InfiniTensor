#include "operators/gather.h"
#include "infini_operators.h"
#include "core/kernel.h"
#include "utils/infiniop_utils.h"

namespace infini {
class GatherOp : public Kernel {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<GatherObj>(_op);

        auto input = op->getInputs(0)->getRawDataPtr<void*>();
        auto index = op->getInputs(1)->getRawDataPtr<void*>();
        auto output = op->getOutput()->getRawDataPtr<void*>();
        // auto axis=op->getOpAttrVector()[1];
        

        if (op->getOpType() == OpType::Gather) {
            CHECK_ERROR(infiniopGather(
                (infiniopGatherDescriptor_t)op->getOpDesc(),
                input, index, output, context->getCurrentStream()));
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

REGISTER_KERNEL(Device::CPU, OpType::Gather, GatherOp, "Gather_infiniop_CPU");
} // namespace infini

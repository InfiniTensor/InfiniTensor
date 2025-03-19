#include "operators/gather.h"
#include "core/kernel.h"
#include "utils/infiniop_utils.h"

namespace infini {
class GatherOp : public Kernel {
    void compute(const Operator &_op, const RuntimeObj *context) const override {
        auto op = as<GatherObj>(_op);
        void *const data = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const indices = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const output = (op->getOutput()->getRawDataPtr<void *>());
        // execute op
        CHECK_ERROR(infiniopGather((infiniopGatherDescriptor_t)op->getOpDesc(),
                                   output, data, indices,
                                   context->getCurrentStream()));
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

REGISTER_KERNEL(Device::CPU, OpType::Gather, GatherOp, "Gather_infiniop_CPU");

} // namespace infini

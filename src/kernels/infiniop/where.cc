#include "operators/where.h"
#include "core/kernel.h"
#include "utils/infiniop_utils.h"

namespace infini {
class WhereOp : public Kernel {
    void compute(const Operator &_op, const RuntimeObj *context) const override {
        auto op = as<WhereObj>(_op);
        void *const xData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const yData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const conData = (op->getInputs(2)->getRawDataPtr<void *>());
        void *const output = (op->getOutput()->getRawDataPtr<void *>());
        // execute op
        CHECK_ERROR(infiniopWhere((infiniopWhereDescriptor_t)op->getOpDesc(),
                                  output, conData, xData, yData,
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

REGISTER_KERNEL(Device::CPU, OpType::Where, WhereOp, "Where_infiniop_CPU");

} // namespace infini

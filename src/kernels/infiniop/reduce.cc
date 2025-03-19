#include "operators/reduce.h"
#include "core/kernel.h"
#include "utils/infiniop_utils.h"

namespace infini {
class ReduceOp : public Kernel {
    void compute(const Operator &_op, const RuntimeObj *context) const override {
        auto op = as<ReduceBaseObj>(_op);
        void *const input = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const output = (op->getOutput(0)->getRawDataPtr<void *>());

        // execute op
        // axes info is stored in create op descriptor, so no need to pass here
        if (op->getOpType() == OpType::ReduceMax) {
            CHECK_ERROR(infiniopReduceMax((infiniopReduceMaxDescriptor_t)op->getOpDesc(),
                                          output, input, nullptr,
                                          context->getCurrentStream()));
        } else if (op->getOpType() == OpType::ReduceMean) {
            CHECK_ERROR(infiniopReduceMean((infiniopReduceMeanDescriptor_t)op->getOpDesc(),
                                           output, input, nullptr,
                                           context->getCurrentStream()));
        } else if (op->getOpType() == OpType::ReduceMin) {
            CHECK_ERROR(infiniopReduceMin((infiniopReduceMinDescriptor_t)op->getOpDesc(),
                                          output, input, nullptr,
                                          context->getCurrentStream()));
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
REGISTER_KERNEL(Device::CPU, OpType::ReduceMax, ReduceOp, "ReduceMax_infiniop_CPU");
REGISTER_KERNEL(Device::CPU, OpType::ReduceMean, ReduceOp, "ReduceMean_infiniop_CPU");
REGISTER_KERNEL(Device::CPU, OpType::ReduceMin, ReduceOp, "ReduceMin_infiniop_CPU");
}

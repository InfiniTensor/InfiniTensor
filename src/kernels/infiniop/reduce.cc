#include "core/kernel.h"
#include "core/op_type.h"
#include "device.h"
#include "operators/reduce.h"
#include "ops/reduce_max/reduce_max.h"
#include "utils/infiniop_utils.h"

namespace infini {

class ReduceOp : public Kernel {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<ReduceBaseObj>(_op);
        void *const x = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const y = (op->getOutput()->getRawDataPtr<void *>());

        if (op->getOpType() == OpType::ReduceMax) {
            CHECK_ERROR(infiniopReduceMax((infiniopReduceMaxDescriptor_t)op->getOpDesc(), y, x, context->getCurrentStream()));
        } else if (op->getOpType() == OpType::ReduceMin) {
            CHECK_ERROR(infiniopReduceMin((infiniopReduceMinDescriptor_t)op->getOpDesc(), y, x, context->getCurrentStream()));
        } else if (op->getOpType() == OpType::ReduceMean) {
            CHECK_ERROR(infiniopReduceMean((infiniopReduceMeanDescriptor_t)op->getOpDesc(), y, x, context->getCurrentStream()));
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

REGISTER_KERNEL(Device::CPU, OpType::ReduceMax, ReduceOp,
                "Reduce_infiniop_cpu");
REGISTER_KERNEL(Device::CPU, OpType::ReduceMin, ReduceOp,
                "Reduce_infiniop_cpu");
REGISTER_KERNEL(Device::CPU, OpType::ReduceMean, ReduceOp,
                "Reduce_infiniop_cpu");
}


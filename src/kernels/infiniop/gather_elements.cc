#include "operators/gather.h"
#include "core/kernel.h"
#include "utils/infiniop_utils.h"

namespace infini {

class GatherElementsOp : public Kernel {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<GatherElementsObj>(_op);
        void *const xData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const indices = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const yData = (op->getOutput()->getRawDataPtr<void *>());
        if (op->getOpType() == OpType::GatherElements) {
            CHECK_ERROR(infiniopGatherElements(
                (infiniopGatherElementsDescriptor_t)op->getOpDesc(),
                yData, xData, indices, context->getCurrentStream()));
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

REGISTER_KERNEL(Device::CUDA, OpType::GatherElements, GatherElementsOp,
                "GatherElements_infiniop_cuda");
REGISTER_KERNEL(Device::CPU, OpType::GatherElements, GatherElementsOp,
                "GatherElements_infiniop_cpu");
}; // namespace infini

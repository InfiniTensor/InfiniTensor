#include "core/kernel.h"
#include "operators/unary.h"
#include "utils/infiniop_utils.h"

namespace infini {

class ClipOp : public Kernel {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<ClipObj>(_op);
        void *const xData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const yData = (op->getOutput()->getRawDataPtr<void *>());

        // execute op
        CHECK_ERROR(infiniopClip((infiniopClipDescriptor_t)op->getOpDesc(),
                                 yData, xData, context->getCurrentStream()));
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

REGISTER_KERNEL(Device::CUDA, OpType::Clip, ClipOp, "Clip_infiniop_CUDA");
REGISTER_KERNEL(Device::CPU, OpType::Clip, ClipOp, "Clip_infiniop_CPU");
}; // namespace infini

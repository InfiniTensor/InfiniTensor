#include "operators/unary.h"
#include "core/kernel.h"
#include "utils/infiniop_utils.h"

namespace infini {

class UnaryOp : public Kernel {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<UnaryObj>(_op);
        void *const xData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const yData = (op->getOutput()->getRawDataPtr<void *>());

        if (op->getOpType() == OpType::Relu) {
            // execute op
            CHECK_ERROR(infiniopRelu((infiniopReluDescriptor_t)op->getOpDesc(),
                                     yData, xData,
                                     context->getCurrentStream()));
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

class ClipOp : public Kernel {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<ClipObj>(_op);
        void *const xData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const yData = (op->getOutput()->getRawDataPtr<void *>());
        IT_ASSERT((op->getOpType() == OpType::Clip));
        if (op->getOpType() == OpType::Clip) {
            // execute op
            CHECK_ERROR(infiniopClip((infiniopClipDescriptor_t)op->getOpDesc(),
                                     yData, xData, 
                                     context->getCurrentStream()));
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

REGISTER_KERNEL(Device::CUDA, OpType::Relu, UnaryOp, "Relu_infiniop_CUDA");
REGISTER_KERNEL(Device::CPU, OpType::Relu, UnaryOp, "Relu_infiniop_CPU");
REGISTER_KERNEL(Device::CUDA, OpType::Clip, ClipOp, "Clip_infiniop_CUDA");
REGISTER_KERNEL(Device::CPU, OpType::Clip, ClipOp, "Clip_infiniop_CPU");
}; // namespace infini

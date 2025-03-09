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
        void *const input = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const output = (op->getOutput()->getRawDataPtr<void *>());
        
        void *min = nullptr;
        void *max = nullptr;
        uint16_t minF16, maxF16;
        float minF32, maxF32;
        if (op->getMin()) {
            if (op->getInputs(0)->getDType() == DataType::Float16) {
                minF16 = float_to_fp16(op->getMin().value());
                min = &minF16;
            }
            if (op->getInputs(0)->getDType() == DataType::Float32) {
                minF32 = op->getMin().value();
                min = &minF32;
            }
        }
        if (op->getMax()) {
            if (op->getInputs(0)->getDType() == DataType::Float16) {
                maxF16 = float_to_fp16(op->getMax().value());
                max = &maxF16;
            }
            if (op->getInputs(0)->getDType() == DataType::Float32) {
                maxF32 = op->getMax().value();
                max = &maxF32;
            }
        }

        CHECK_ERROR(infiniopClip((infiniopClipDescriptor_t)op->getOpDesc(),
                                 output, input, min, max,
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

REGISTER_KERNEL(Device::CUDA, OpType::Relu, UnaryOp, "Relu_infiniop_CUDA");
REGISTER_KERNEL(Device::CPU, OpType::Relu, UnaryOp, "Relu_infiniop_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Clip, ClipOp, "Clip_infiniop_CPU");
}; // namespace infini

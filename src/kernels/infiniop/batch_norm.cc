#include "operators/batch_norm.h"
#include "core/kernel.h"
#include "utils/infiniop_utils.h"

namespace infini {

class BatchNormOp : public Kernel {
    void compute(const Operator &_op, const RuntimeObj *context) const override {
        auto op = as<BatchNormObj>(_op);
        void *const yData = (op->getOutput()->getRawDataPtr<void *>());
        void *const xData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const meanData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const varData = (op->getInputs(2)->getRawDataPtr<void *>());
        void *const scaleData = (op->getInputs(3)->getRawDataPtr<void *>());
        void *const biasData = (op->getInputs(4)->getRawDataPtr<void *>());
       

        CHECK_ERROR(infiniopBatchNorm(
            (infiniopBatchNormDescriptor_t)op->getOpDesc(), yData, xData, scaleData, biasData,
            meanData, varData, context->getCurrentStream()));
    }

    PerfRecord tune(const Operator &op, const RuntimeObj *context) const override {
        // TODO: tune should be in infiniop
        return PerfRecord();
    }

    void compute(const Operator &op, const PerfRecord &record,
                const RuntimeObj *context) const override {
        compute(op, context);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::BatchNormalization, BatchNormOp, "BatchNorm_infiniop_CUDA");
REGISTER_KERNEL(Device::CPU, OpType::BatchNormalization, BatchNormOp, "BatchNorm_infiniop_CPU");
}; // namespace infini
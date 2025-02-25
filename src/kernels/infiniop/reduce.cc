#include "operators/reduce.h"
#include "core/kernel.h"
#include "utils/infiniop_utils.h"

namespace infini {
class ReduceOp : public Kernel{
    void compute(const Operator &_op, const RuntimeObj *context) const override {
            auto op = as<ReduceBaseObj>(_op);
            void *const yData = (op->getOutput()->getRawDataPtr<void *>());
            void *const xData = (op->getInputs(0)->getRawDataPtr<void *>());
            
            if (op->getOpType() == OpType::ReduceMax){
                CHECK_ERROR(infiniopReducemax(
                    (infiniopReducemaxDescriptor_t)op->getOpDesc(), yData, xData, nullptr, 0, context->getCurrentStream()));
            }else if (op->getOpType() == OpType::ReduceMean){
                CHECK_ERROR(infiniopReducemean(
                    (infiniopReducemeanDescriptor_t)op->getOpDesc(), yData, xData, nullptr, 0, context->getCurrentStream()));
            }else if (op->getOpType() == OpType::ReduceMin){
                CHECK_ERROR(infiniopReducemin(
                    (infiniopReduceminDescriptor_t)op->getOpDesc(), yData, xData, nullptr, 0, context->getCurrentStream()));
            }else{
                IT_TODO_HALT();
            }
    }
    PerfRecord tune(const Operator &op,
                    const RuntimeObj *context) const override {
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
};
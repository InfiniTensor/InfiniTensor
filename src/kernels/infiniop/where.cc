#include "operators/where.h"
#include "core/kernel.h"
#include "utils/infiniop_utils.h"

namespace infini {
    class WhereOp : public Kernel{
        void compute(const Operator &_op, const RuntimeObj *context) const override {
                auto op = as<WhereObj>(_op);
                void *const yData = (op->getOutput()->getRawDataPtr<void *>());
                void *const InputxData = (op->getInputs(0)->getRawDataPtr<void *>());
                void *const InputyData = (op->getInputs(1)->getRawDataPtr<void *>());
                void *const conditionData = (op->getInputs(2)->getRawDataPtr<void *>());
                
                CHECK_ERROR(infiniopWhere(
                    (infiniopWhereDescriptor_t)op->getOpDesc(), yData, InputxData, InputyData, conditionData, context->getCurrentStream()));
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
    REGISTER_KERNEL(Device::CPU, OpType::Where, WhereOp, "Where_infiniop_CPU");
    REGISTER_KERNEL(Device::CUDA, OpType::Where, WhereOp, "Where_infiniop_CUDA");
}
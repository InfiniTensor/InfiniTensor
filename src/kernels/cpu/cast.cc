#include "core/kernel.h"
#include "operators/unary.h"

namespace infini{

class CastCPU final : public CpuKernelWithoutConfig {
    template <typename InputT, typename OutputT>
    void doCompute(const Operator &_op) const {
        auto op = as<CastObj>(_op);
        IT_ASSERT(op != nullptr);

        const auto *input = op->getInputs(0)->getRawDataPtr<const InputT *>();
        auto *output = op->getOutput()->getRawDataPtr<OutputT *>();

        for (size_t i = 0; i < op->getOutput()->size(); ++i) {
            output[i] = static_cast<OutputT>(input[i]);
        }
    }

    void compute(const Operator &_op, const RuntimeObj *context) const override{
        auto op = as<CastObj>(_op);
        IT_ASSERT(op != nullptr);

        switch (op->getType()) {
        case CastType::Int322Int64:
            doCompute<int32_t, int64_t>(_op);
            break;
        case CastType::Int642Int32:
            doCompute<int64_t, int32_t>(_op);
            break;
        case CastType::Float2Int32:
            doCompute<float, int32_t>(_op);
            break;
        case CastType::Float2Int64:
            doCompute<float, int64_t>(_op);
            break;
        case CastType::Int322Float:
            doCompute<int32_t, float>(_op);
            break;
        case CastType::Int642Float:
            doCompute<int64_t, float>(_op);
            break;
        case CastType::Float2Float:
            doCompute<float, float>(_op);
            break;
        default:
            IT_ASSERT(false, 
                "Cast type not supported") ; 
        }
    }
};

REGISTER_KERNEL(Device::CPU, OpType::Cast, CastCPU, "Cast_CPU");

}
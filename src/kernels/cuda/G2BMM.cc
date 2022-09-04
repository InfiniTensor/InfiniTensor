#include "operators/conv.h"
#include "core/kernel.h"
#include "cuda/cuda_runtime.h"
#include "custom_ops.h"
#include <chrono>
#include <functional>
#include <tuple>
namespace infini {


class G2BMMCudnn : public Kernel {

    void compute(const Operator &op,  const RuntimeObj *context) const override {
        PerfRecord record;
        compute(op, record, context);
    }

    PerfRecord tune(const Operator &_op,
                    const RuntimeObj *_context) const override {
        PerfRecord record;
    }
    void compute(const Operator &_op, const PerfRecord &_record,
                const RuntimeObj *_context) const override {
        
    }

};

REGISTER_KERNEL(Device::CUDA, OpType::G2BMM, DataType::Float32, G2BMMCudnn,
                "G2BMM_cuDNN_CUDA_Float32");

REGISTER_KERNEL(Device::CUDA, OpType::G2BMM, DataType::UInt32, G2BMMCudnn,
                "G2BMM_cuDNN_CUDA_UInt32");

} // namespace infini

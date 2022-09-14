#include "operators/unary.h"
#include "core/kernel.h"
#include "cuda/cuda_unary.h"
#include "cuda/cuda_runtime.h"

namespace infini {

class UnaryCuda : public Kernel {
    void compute(const Operator &_op, const PerfRecord &record,
                 const RuntimeObj *_context) const override {
        unary_kernel(_op);
    }

    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        compute(_op, {}, _context);
    }
    // Premise: op is idempotent since it is called multiple times.
    PerfRecord tune(const Operator &_op,
                    const RuntimeObj *_context) const override {
        PerfRecord ret;
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);
        ret.time = timeit([&]() { compute(_op, _context); },
                          [&]() { context->sync(); });
        return ret;
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Softmax, DataType::Float32, UnaryCuda,
                "Softmax_CUDA_Float32");
REGISTER_KERNEL(Device::CUDA, OpType::Relu, DataType::Float32, UnaryCuda,
                "Relu_CUDA_Float32");
REGISTER_KERNEL(Device::CUDA, OpType::Sigmoid, DataType::Float32, UnaryCuda,
                "Sigmoid_CUDA_Float32");
REGISTER_KERNEL(Device::CUDA, OpType::Tanh, DataType::Float32, UnaryCuda,
                "Tanh_CUDA_Float32");
REGISTER_KERNEL(Device::CUDA, OpType::Abs, DataType::Float32, UnaryCuda,
                "Abs_CUDA_Float32");
}; // namespace infini

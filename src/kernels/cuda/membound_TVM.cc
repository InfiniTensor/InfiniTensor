#include "core/kernel.h"
#include "cuda/cuda_runtime.h"
#include "operators/membound.h"
#include "operators/pooling.h"

namespace infini {

class TVMRecord : public PerfRecord {
    // TODO: Add more attrs
};

class MemboundTVM : public Kernel {
    void compute(const Operator &_op, const PerfRecord &record,
                 const RuntimeObj *_context) const override {
        auto op = as<MemBoundObj>(_op);
        // auto context = dynamic_cast<const CudaRuntimeObj *>(_context);
        // cudnnStatus_t stat;
        // void *const inData = (op->getInputs(0)->getRawDataPtr<void *>());
        // void *const outData = (op->getOutput()->getRawDataPtr<void *>());
    }

    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        IT_ASSERT(false, "A TVM record is required for membound kernel.");
    }

    // Premise: op is idempotent since it is called multiple times.
    PerfRecord tune(const Operator &_op,
                    const RuntimeObj *_context) const override {
        PerfRecord ret;
        auto op = as<MemBoundObj>(_op);
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);
        // TODO: invoke Ansor to tune a membound kernel
        // Evaluate the kernel
        ret.time = timeit(
            [&]() {
                // TODO: run the kernel
            },
            [&]() { context->sync(); });
        return ret;
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::MemBound, DataType::Float32, MemboundTVM,
                "Memobund_TVM_Ansor");
}; // namespace infini
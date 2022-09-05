#include "operators/GBMML.h"
#include "core/kernel.h"
#include "cuda/cuda_runtime.h"
#include "custom_ops.h"
#include <chrono>
#include <functional>
#include <tuple>

namespace infini {

class GBMMLCudnn : public Kernel {

    bool cuDNNCustomOp(const Ref<GBMMLObj> &op, const CudaRuntimeObj *context) const {
        float *const inAData = (op->getInputs(0)->getRawDataPtr<float *>());
        float *const inBData = (op->getInputs(1)->getRawDataPtr<float *>());
        if(op->getInputs().size() > 2)
            IT_TODO_HALT();

        float *const outData = (op->getOutput()->getRawDataPtr<float *>());
        
        const auto [b, m, w, n, dilation] = op->getBMWND();

        _sgbmml(inAData, inBData, outData, b, m, n, w, dilation); 
        checkCudaError(cudaDeviceSynchronize());
        return true;
    }
    void compute(const Operator &op,  const RuntimeObj *context) const override {
        PerfRecord record;
        compute(op, record, context);
    }

    PerfRecord tune(const Operator &_op,
                    const RuntimeObj *_context) const override {
        PerfRecord record;
        auto op = as<GBMMLObj>(_op);
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);

        record.time = std::numeric_limits<double>::max();
        const auto [warmupRounds, timingRounds]= op->getB() > 100 ?
                                                 tuple{1, 3}: tuple{5, 15};
        double tmp = timeit([&]() { cuDNNCustomOp(op, context); },
                            [&]() { context->sync(); },
                            warmupRounds, timingRounds);
        if (tmp < record.time)
            record.time = tmp;
        IT_ASSERT(record.time < std::numeric_limits<double>::max(), "Error occured "
                                                                 "during runtime");
        return record;
    }
    
    void compute(const Operator &_op, const PerfRecord &_record,
                const RuntimeObj *_context) const override {
        auto op = as<GBMMLObj>(_op);
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);
        bool success = cuDNNCustomOp(op, context);
        IT_ASSERT(success);
    }
    
};

REGISTER_KERNEL(Device::CUDA, OpType::GBMML, DataType::Float32, GBMMLCudnn,
                "GBMML_cuDNN_CUDA_Float32");

} // namespace infini
#include "operators/GBMM.h"
#include "cuda/cuda_kernel_without_config.h"
#include "cuda/cuda_runtime.h"
#include "cuda/gbmm_g2bmm.h"
#include <chrono>
#include <functional>
#include <tuple>

namespace infini {

class GBMMCudnn : public CudaKernelWithoutConfig {

    bool gbmmKernel(const Ref<GBMMObj> &op,
                    const CudaRuntimeObj *context) const {
        if (op->getInputs().size() > 2)
            IT_TODO_HALT();

        const auto [b, m, w, n, dilation] = op->getBMWND();
        // printf("%d %d %d %d %d\n", b, m, n, w, dilation);
        if (op->getDType() == DataType::Float16)
            _sgbmml(op->getInputs(0)->getRawDataPtr<half *>(),
                    op->getInputs(1)->getRawDataPtr<half *>(),
                    op->getOutput()->getRawDataPtr<half *>(), b, m, n, w,
                    dilation);
        else
            _sgbmml(op->getInputs(0)->getRawDataPtr<float *>(),
                    op->getInputs(1)->getRawDataPtr<float *>(),
                    op->getOutput()->getRawDataPtr<float *>(), b, m, n, w,
                    dilation);
        // checkCudaError(cudaDeviceSynchronize());
        return true;
    }

    PerfRecord tune(const Operator &_op,
                    const RuntimeObj *_context) const override {
        auto op = as<GBMMObj>(_op);
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);

        auto record =
            make_ref<PerfRecordObj>(std::numeric_limits<double>::max());
        const auto [warmupRounds, timingRounds] =
            op->getB() > 100 ? tuple{1, 3} : tuple{5, 15};
        double tmp =
            timeit([&]() { gbmmKernel(op, context); },
                   [&]() { context->sync(); }, warmupRounds, timingRounds);
        if (tmp < record->time)
            record->time = tmp;
        IT_ASSERT(record->time < std::numeric_limits<double>::max(),
                  "Error occured "
                  "during runtime");
        return record;
    }

    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<GBMMObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32 ||
                  op->getDType() == DataType::Float16);
        IT_ASSERT(op->getInputs(1)->getDType() == op->getDType());
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);
        bool success = gbmmKernel(op, context);
        IT_ASSERT(success);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::GBMM, GBMMCudnn, "GBMM_cuDNN_CUDA");

} // namespace infini

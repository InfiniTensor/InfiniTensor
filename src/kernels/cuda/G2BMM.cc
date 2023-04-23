#include "operators/G2BMM.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"
#include "cuda/gbmm_g2bmm.h"
#include <chrono>
#include <functional>
#include <tuple>
namespace infini {

class G2BMMCudnn : public CudaKernelWithoutConfig {

    bool g2bmmKernel(const Ref<G2BMMObj> &op,
                     const CudaRuntimeObj *context) const {
        float *const inAData = (op->getInputs(0)->getRawDataPtr<float *>());
        float *const inBData = (op->getInputs(1)->getRawDataPtr<float *>());
        if (op->getInputs().size() > 2)
            IT_TODO_HALT();

        float *const outData = (op->getOutput()->getRawDataPtr<float *>());

        const auto [b, n, m, width, dilation] = op->getBMKWD();

        _sg2bmm(inAData, inBData, outData, b, n, m, width, dilation);
        // checkCudaError(cudaDeviceSynchronize());
        return true;
    }

    PerfRecord tune(const Operator &_op,
                    const RuntimeObj *_context) const override {
        auto op = as<G2BMMObj>(_op);
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);

        auto record =
            make_ref<PerfRecordObj>(std::numeric_limits<double>::max());
        const auto [warmupRounds, timingRounds] =
            op->getB() > 100 ? tuple{1, 1} : tuple{1, 2};
        double tmp =
            timeit([&]() { g2bmmKernel(op, context); },
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
        auto op = as<G2BMMObj>(_op);
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);
        bool success = g2bmmKernel(op, context);
        IT_ASSERT(success);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::G2BMM, DataType::Float32, G2BMMCudnn,
                "G2BMM_cuDNN_CUDA_Float32");

} // namespace infini

#include "operators/det.h"
#include "cuda/cuda_det.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"

namespace infini {

class DetCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<DetObj>(_op);
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);
        void *const inputData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const outputData = (op->getOutput()->getRawDataPtr<void *>());
        const auto batch_size = op->getOutput(0)->getDims()[0];
        const auto n = op->getInputs(0)->getDims().at(1);
        auto dataType = op->getDType();

        // Data type other than float32 needs additional support
        // See det_kernel() in det.cu for details
        if (dataType != DataType::Float32) {
            IT_TODO_HALT();
        }

        det_kernel(context, inputData, outputData, n, batch_size,
                   static_cast<int>(op->getMode()));
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Det, DetCuda, "Det_CUDA");

}; // namespace infini

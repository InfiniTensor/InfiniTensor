#include "operators/softmax.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"
#include "cuda/softmax.h"

namespace infini {
class SoftmaxCudnn : public CudaKernelWithoutConfig {

    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<SoftmaxObj>(_op);
        auto x = op->getInputs(0)->getRawDataPtr<float *>();
        auto y = op->getOutput(0)->getRawDataPtr<float *>();
        auto dims = op->getInputs(0)->getDims();

        int batch_size = 1;
        for (size_t i = 0; i < dims.size(); ++i)
            batch_size *= dims[i];
        int dim = dims[op->getAxis()];

        int block_num = batch_size / dim;
        int max_threadblock_size = batch_size / block_num;
        softmax_kernel(max_threadblock_size, block_num, x, y, dim,
                       op->getInputs(0)->getStride().at(op->getAxis()));
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Softmax, DataType::Float32, SoftmaxCudnn,
                "Softmax_CUDA_Float32");
} // namespace infini

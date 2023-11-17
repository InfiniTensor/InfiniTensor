#include "operators/layer_norm.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_layernorm.h"
#include "cuda/cuda_runtime.h"

namespace infini {

class LayerNormCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<LayerNormObj>(_op);

        void *const inputData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const scaleData = (op->getInputs(1)->getRawDataPtr<void *>());

        void *const outputData = (op->getOutput()->getRawDataPtr<void *>());
        const auto &opOutputShape = op->getOutput()->getDims();

        float eps = op->getEps();
        const int axis = op->getAxis();
        const int stride = op->getInputs(0)->getStride().at(axis);

        auto dims = op->getInputs(0)->getDims();
        int dimsize = dims[op->getAxis()];
        int size = op->getOutput(0)->size();
        int scaleSize = op->getInputs(1)->size();
        if (op->numInputs() == 3) {
            void *const biasData = (op->getInputs(2)->getRawDataPtr<void *>());
            int biasSize = op->getInputs(2)->size();
            // printf("kernel bias:true:%d\n", 1);
            hasLaynormKernel((float *)inputData, (float *)scaleData, eps, size,
                             scaleSize, dimsize, stride, (float *)outputData,
                             (float *)biasData, biasSize);
        } else {
            // printf("kernel bias:false:%d\n", 0);
            LaynormKernel((float *)inputData, (float *)scaleData, eps, size,
                          scaleSize, dimsize, stride, (float *)outputData);
        }
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::LayerNormalization, DataType::Float32,
                LayerNormCuda, "LayerNorm_CUDA_Float32");

}; // namespace infini

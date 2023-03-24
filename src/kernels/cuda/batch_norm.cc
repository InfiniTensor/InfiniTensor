#include "operators/batch_norm.h"
#include "core/kernel.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"

namespace infini {
class BatchNormCudnn : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<BatchNormObj>(_op);
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);
        cudnnStatus_t stat;
        void *const inData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const outData = (op->getOutput()->getRawDataPtr<void *>());
        void *const meanData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const varData = (op->getInputs(2)->getRawDataPtr<void *>());
        void *const scaleData = (op->getInputs(3)->getRawDataPtr<void *>());
        void *const biasData = (op->getInputs(4)->getRawDataPtr<void *>());

        auto dims = op->getInputs(0)->getDims();
        // Only 4D and 5D tensors are supported by
        // cudnnBatchNormalizationForwardInference
        IT_ASSERT(dims.size() == 4);

        int dimArray[4], strideArray[4], dimPArray[4], stridePArray[4];
        for (size_t i = 0; i < dims.size(); ++i) {
            dimArray[i] = dims[i];
            strideArray[i] = op->getInputs(0)->getStride()[i];
            dimPArray[i] = 1;
            stridePArray[i] = 1;
        }
        dimPArray[1] = op->getInputs(1)->getDims()[0];
        stridePArray[0] = op->getInputs(1)->getDims()[0];
        // get inputs
        cudnnTensorDescriptor_t inDesc;
        checkCudnnError(cudnnCreateTensorDescriptor(&inDesc));
        checkCudnnError(cudnnSetTensorNdDescriptor(
            inDesc, CUDNN_DATA_FLOAT, dims.size(), dimArray, strideArray));

        // get bnScaleBiasMeanVarDesc
        cudnnTensorDescriptor_t paraDesc;
        checkCudnnError(cudnnCreateTensorDescriptor(&paraDesc));
        checkCudnnError(cudnnSetTensorNdDescriptor(
            paraDesc, CUDNN_DATA_FLOAT, dims.size(), dimPArray, stridePArray));

        float alpha = 1.f, beta = 0.f;
        // This mode is intended for use after convolutional layers
        stat = cudnnBatchNormalizationForwardInference(
            context->cudnnHandle(), CUDNN_BATCHNORM_SPATIAL, &alpha, &beta,
            inDesc, inData, inDesc, outData, paraDesc, scaleData, biasData,
            meanData, varData, op->getEps());
        if (stat != CUDNN_STATUS_SUCCESS)
            return;

        // Destories in CUDA does not require sync. But cuDNN does not state
        // whether sync is required before destories.
        checkCudnnError(cudnnDestroyTensorDescriptor(inDesc));
        checkCudnnError(cudnnDestroyTensorDescriptor(paraDesc));
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::BatchNorm, DataType::Float32,
                BatchNormCudnn, "BatchNorm_cuDNN_CUDA_Float32");
} // namespace infini

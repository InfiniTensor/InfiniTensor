#include "operators/layer_norm.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"

namespace infini {
class LayerNormCudnn : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<LayerNormObj>(_op);
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);
        cudnnStatus_t stat;
        void *const inData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const outData = (op->getOutput()->getRawDataPtr<void *>());
        void *const scaleData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const biasData = (op->getInputs(2)->getRawDataPtr<void *>());

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

        // get lnScaleBiasDesc
        // 是否正确？
        cudnnTensorDescriptor_t paraDesc;
        checkCudnnError(cudnnCreateTensorDescriptor(&paraDesc));
        checkCudnnError(cudnnSetTensorNdDescriptor(
            paraDesc, CUDNN_DATA_FLOAT, dims.size(), dimPArray, stridePArray));

        // 用于计算 dstValue = alpha[0]*resultValue + beta[0]*priorDstValue
        float alpha = 1.f, beta = 0.f;
        int groupCnt = 1;
        // This mode is intended for use after convolutional layers
        // 为什么 out desc 要用 inDesc？
        stat = cudnnNormalizationForwardInference(
            context->cudnnHandle(), CUDNN_NORM_PER_ACTIVATION,
            CUDNN_NORM_OPS_NORM, CUDNN_NORM_ALGO_STANDARD, &alpha, &beta,
            inDesc, inData, paraDesc, scaleData, biasData, inDesc, outData,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, op->getEps(),
            groupCnt);
        if (stat != CUDNN_STATUS_SUCCESS)
            return;

        // Destories in CUDA does not require sync. But cuDNN does not state
        // whether sync is required before destories.
        checkCudnnError(cudnnDestroyTensorDescriptor(inDesc));
        checkCudnnError(cudnnDestroyTensorDescriptor(paraDesc));
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::LayerNormalization, DataType::Float32,
                LayerNormCudnn, "LayerNorm_cuDNN_CUDA_Float32");
} // namespace infini

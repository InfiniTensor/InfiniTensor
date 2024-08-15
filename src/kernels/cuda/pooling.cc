#include "operators/pooling.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"

namespace infini {
class poolingCudnn : public CudaKernelWithoutConfig {
    virtual cudnnPoolingMode_t getPoolingMode() const = 0;
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<PoolingObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);
        void *const inData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const outData = (op->getOutput()->getRawDataPtr<void *>());

        const auto [n, c, h, w, kh, kw] = op->getNCHWRS();
        const auto [ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
        auto inDims = op->getInputs(0)->getDims();
        auto outDims = op->getOutput()->getDims();

        // Create and set tensor descriptor for input
        cudnnTensorDescriptor_t inDesc;
        checkCudnnError(cudnnCreateTensorDescriptor(&inDesc));
        checkCudnnError(cudnnSetTensor4dDescriptor(
            inDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));

        // Get outputs
        int outn = outDims.at(0), outc = outDims.at(1),
            outh = outDims.size() == 3 ? 1 : outDims.at(2),
            outw = outDims.size() == 3 ? outDims.at(2) : outDims.at(3);
        // // NOTICE: cudnn pooling does not support ceil mode, so the shape
        // // inference of cudnn pooling is not consistant with our
        // framework. Ceil
        // // mode is also supported in Pytorch and ONNX. See
        // //
        // https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d
        // // and
        // https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxPool
        // // for reference.
        // // TODO: Make sure the result after considering ceil mode is
        // correct.
        // // int outn, outc, outh, outw;
        // // checkCudnnError(cudnnGetPooling2dForwardOutputDim(poolingDesc,
        // // inDesc, &outn, &outc, &outh, &outw));
        cudnnTensorDescriptor_t outDesc;
        checkCudnnError(cudnnCreateTensorDescriptor(&outDesc));
        checkCudnnError(cudnnSetTensor4dDescriptor(outDesc, CUDNN_TENSOR_NCHW,
                                                   CUDNN_DATA_FLOAT, outn, outc,
                                                   outh, outw));

        // Create and set pooling descriptor for average pooling
        cudnnPoolingDescriptor_t poolingDesc;
        checkCudnnError(cudnnCreatePoolingDescriptor(&poolingDesc));
        checkCudnnError(
            cudnnSetPooling2dDescriptor(poolingDesc,
                                        getPoolingMode(), // Pooling mode
                                        CUDNN_NOT_PROPAGATE_NAN,
                                        kh, // pooling window height
                                        kw, // pooling window width
                                        ph, // vertical padding
                                        pw, // horizontal padding
                                        sh, // vertical Stride
                                        sw  // horizontal stride
                                        ));

        // Perform the pooling operation
        float alpha = 1.0f, beta = 0.0f;
        checkCudnnError(cudnnPoolingForward(context->cudnnHandle(), poolingDesc,
                                            &alpha, inDesc, inData, &beta,
                                            outDesc, outData));

        // Destories in CUDA does not require sync. But cuDNN does not state
        // whether sync is required before destories.
        checkCudnnError(cudnnDestroyTensorDescriptor(inDesc));
        checkCudnnError(cudnnDestroyTensorDescriptor(outDesc));
        checkCudnnError(cudnnDestroyPoolingDescriptor(poolingDesc));
    }
};

class maxPoolCudnn : public poolingCudnn {
    cudnnPoolingMode_t getPoolingMode() const override {
        return CUDNN_POOLING_MAX;
    }
};

class avgPoolCudnn : public poolingCudnn {
    cudnnPoolingMode_t getPoolingMode() const override {
        return CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::MaxPool, maxPoolCudnn,
                "MaxPool_cuDNN_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::AveragePool, avgPoolCudnn,
                "AvgPool_cuDNN_CUDA");

}; // namespace infini

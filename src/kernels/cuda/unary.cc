#include "operators/unary.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_unary.h"

namespace infini {

class UnaryCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        unary_kernel(_op);
    }
};

class ActivationCudnn : public CudaKernelWithoutConfig {
    virtual cudnnActivationMode_t getOpType() const = 0;
    virtual tuple<float, float> getAlphBeta() const { return {1.f, 0.f}; }
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);

        void *const inputData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const outputData = (op->getOutput()->getRawDataPtr<void *>());

        cudnnTensorDescriptor_t inputDesc, outputDesc;
        auto dim = op->getInputs(0)->getDims();
        // assume input and output have the same strides.
        auto stride = op->getInputs(0)->getStride();
        // CUDNN requires that dim >= 4.
        while (dim.size() < 4)
            dim.push_back(1);
        while (stride.size() < 4)
            stride.push_back(1);

        // get inputs
        checkCudnnError(cudnnCreateTensorDescriptor(&inputDesc));
        checkCudnnError(cudnnSetTensorNdDescriptor(inputDesc, CUDNN_DATA_FLOAT,
                                                   dim.size(), dim.data(),
                                                   stride.data()));

        // get outputs
        checkCudnnError(cudnnCreateTensorDescriptor(&outputDesc));
        checkCudnnError(cudnnSetTensorNdDescriptor(outputDesc, CUDNN_DATA_FLOAT,
                                                   dim.size(), dim.data(),
                                                   stride.data()));

        // get op descriptor
        cudnnActivationDescriptor_t activationDesc;
        checkCudnnError(cudnnCreateActivationDescriptor(&activationDesc));
        checkCudnnError(cudnnSetActivationDescriptor(
            activationDesc, getOpType(), CUDNN_NOT_PROPAGATE_NAN, 0.0));

        auto [alpha, beta] = getAlphBeta();
        cudnnStatus_t stat = cudnnActivationForward(
            context->cudnnHandle(), activationDesc, &alpha, inputDesc,
            inputData, &beta, outputDesc, outputData);
        if (stat != CUDNN_STATUS_SUCCESS)
            return;

        // Destories in CUDA does not require sync. But cuDNN does not state
        // whether sync is required before destories.
        checkCudnnError(cudnnDestroyActivationDescriptor(activationDesc));
        checkCudnnError(cudnnDestroyTensorDescriptor(outputDesc));
        checkCudnnError(cudnnDestroyTensorDescriptor(inputDesc));
    }
};

class SoftmaxCudnn : public CudaKernelWithoutConfig {
    virtual cudnnSoftmaxAlgorithm_t getAlgorithmType() const = 0;
    virtual cudnnSoftmaxMode_t getModeType() const = 0;
    virtual tuple<float, float> getAlphBeta() const { return {1.f, 0.f}; }
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);

        void *const inputData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const outputData = (op->getOutput()->getRawDataPtr<void *>());

        cudnnTensorDescriptor_t inputDesc, outputDesc;
        auto dim = op->getInputs(0)->getDims();
        if (dim.size() > 4)
            IT_TODO_HALT();
        int dim_array[4] = {1, 1, 1, 1};
        memcpy(dim_array + (4 - dim.size()), dim.data(),
               dim.size() * sizeof(int));

        // get inputs
        checkCudnnError(cudnnCreateTensorDescriptor(&inputDesc));
        checkCudnnError(cudnnSetTensor4dDescriptor(
            inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, dim_array[0],
            dim_array[1], dim_array[2], dim_array[3]));

        // get outputs
        checkCudnnError(cudnnCreateTensorDescriptor(&outputDesc));
        checkCudnnError(cudnnSetTensor4dDescriptor(
            outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, dim_array[0],
            dim_array[1], dim_array[2], dim_array[3]));

        auto [alpha, beta] = getAlphBeta();
        cudnnStatus_t stat = cudnnSoftmaxForward(
            context->cudnnHandle(), getAlgorithmType(), getModeType(), &alpha,
            inputDesc, inputData, &beta, outputDesc, outputData);
        if (stat != CUDNN_STATUS_SUCCESS)
            return;

        // Destories in CUDA does not require sync. But cuDNN does not state
        // whether sync is required before destories.
        checkCudnnError(cudnnDestroyTensorDescriptor(inputDesc));
        checkCudnnError(cudnnDestroyTensorDescriptor(outputDesc));
    }
};

class ReluCudnn : public ActivationCudnn {
    cudnnActivationMode_t getOpType() const override {
        return CUDNN_ACTIVATION_RELU;
    }
};

class SigmoidCudnn : public ActivationCudnn {
    cudnnActivationMode_t getOpType() const override {
        return CUDNN_ACTIVATION_SIGMOID;
    }
};

class TanhCudnn : public ActivationCudnn {
    cudnnActivationMode_t getOpType() const override {
        return CUDNN_ACTIVATION_TANH;
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Relu, DataType::Float32, ReluCudnn,
                "Relu_CUDA_Float32");
REGISTER_KERNEL(Device::CUDA, OpType::Sigmoid, DataType::Float32, SigmoidCudnn,
                "Sigmoid_CUDA_Float32");
REGISTER_KERNEL(Device::CUDA, OpType::HardSigmoid, DataType::Float32, UnaryCuda,
                "Hard_Sigmoid_CUDA_Float32");
REGISTER_KERNEL(Device::CUDA, OpType::HardSwish, DataType::Float32, UnaryCuda,
                "Hard_Swish_CUDA_Float32");
REGISTER_KERNEL(Device::CUDA, OpType::Tanh, DataType::Float32, TanhCudnn,
                "Tanh_CUDA_Float32");
REGISTER_KERNEL(Device::CUDA, OpType::Abs, DataType::Float32, UnaryCuda,
                "Abs_CUDA_Float32");
REGISTER_KERNEL(Device::CUDA, OpType::Sqrt, DataType::Float32, UnaryCuda,
                "Sqrt_CUDA_Float32");
REGISTER_KERNEL(Device::CUDA, OpType::Gelu, DataType::Float32, UnaryCuda,
                "Gelu_CUDA_Float32");
REGISTER_KERNEL(Device::CUDA, OpType::Neg, DataType::Float32, UnaryCuda,
                "Neg_CUDA_Float32");
REGISTER_KERNEL(Device::CUDA, OpType::Erf, DataType::Float32, UnaryCuda,
                "Erf_CUDA_Float32");

// REGISTER_KERNEL(Device::CUDA, OpType::Softmax, DataType::Float32, UnaryCuda,
//                 "Softmax_CUDA_Float32");
// REGISTER_KERNEL(Device::CUDA, OpType::Relu, DataType::Float32, UnaryCuda,
//                 "Relu_CUDA_Float32");
// REGISTER_KERNEL(Device::CUDA, OpType::Sigmoid, DataType::Float32, UnaryCuda,
//                 "Sigmoid_CUDA_Float32");
// REGISTER_KERNEL(Device::CUDA, OpType::Tanh, DataType::Float32, UnaryCuda,
//                 "Tanh_CUDA_Float32");
// REGISTER_KERNEL(Device::CUDA, OpType::Abs, DataType::Float32, UnaryCuda,
//                 "Abs_CUDA_Float32");
}; // namespace infini

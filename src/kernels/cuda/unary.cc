#include "operators/unary.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_unary.h"
#include "cuda/cuda_utility.h"

namespace infini {

class UnaryCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        unary_kernel(_op);
    }
};

class EluCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<EluObj>(_op);

        void *const inputData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const outputData = (op->getOutput()->getRawDataPtr<void *>());

        size_t size = op->getInputs(0)->size();
        elu_kernel((float *)inputData, (float *)outputData, size,
                   op->getAlpha());
    }
};

class CastCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<CastObj>(_op);

        size_t num = op->getOutput()->size();
        void *const inputData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const outputData = (op->getOutput()->getRawDataPtr<void *>());

        if (op->getType() == CastType::Float162Float) {
            IT_ASSERT(op->getDType() == DataType::Float16 &&
                      op->getOutDType() == DataType::Float32);
            cast_kernel<half, float>((half *)inputData, (float *)outputData,
                                     num);
        } else if (op->getType() == CastType::Float2Float16) {
            IT_ASSERT(op->getDType() == DataType::Float32 &&
                      op->getOutDType() == DataType::Float16);
            cast_kernel<float, half>((float *)inputData, (half *)outputData,
                                     num);
        } else if (op->getType() == CastType::Float2Int32) {
            IT_ASSERT(op->getDType() == DataType::Float32 &&
                      op->getOutDType() == DataType::Int32);
            cast_kernel<float, int32_t>((float *)inputData,
                                        (int32_t *)outputData, num);
        } else if (op->getType() == CastType::Float2Int8) {
            IT_ASSERT(op->getDType() == DataType::Float32 &&
                      op->getOutDType() == DataType::Int8);
            cast_kernel<float, int8_t>((float *)inputData, (int8_t *)outputData,
                                       num);
        } else if (op->getType() == CastType::Int82Float) {
            IT_ASSERT(op->getDType() == DataType::Int8 &&
                      op->getOutDType() == DataType::Float32);
            cast_kernel<int8_t, float>((int8_t *)inputData, (float *)outputData,
                                       num);
        }else if (op->getType() == CastType::Float2Bool) {
            IT_ASSERT(op->getDType() == DataType::Float32 &&
                      op->getOutDType() == DataType::Bool);
            cast_kernel<float, bool>((float *)inputData, (bool *)outputData,
                                     num);
        } else if (op->getType() == CastType::Int642Int32) {
            IT_ASSERT(op->getDType() == DataType::Int64 &&
                      op->getOutDType() == DataType::Int32);
            cast_kernel<int64_t, int32_t>((int64_t *)inputData,
                                          (int32_t *)outputData, num);
        } else if (op->getType() == CastType::Int322Int64) {
            IT_ASSERT(op->getDType() == DataType::Int32 &&
                      op->getOutDType() == DataType::Int64);
            cast_kernel<int32_t, int64_t>((int32_t *)inputData,
                                          (int64_t *)outputData, num);
        } else if (op->getType() == CastType::Int642Float) {
            IT_ASSERT(op->getDType() == DataType::Int64 &&
                      op->getOutDType() == DataType::Float32);
            cast_kernel<int64_t, float>((int64_t *)inputData,
                                        (float *)outputData, num);
        } else if (op->getType() == CastType::Bool2Int32) {
            IT_ASSERT(op->getDType() == DataType::Bool &&
                      op->getOutDType() == DataType::Int32);
            cast_kernel<bool, int32_t>((bool *)inputData, (int32_t *)outputData,
                                       num); 
        }
        else {
            IT_ASSERT(false);
        }
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

        auto cudnnDataType = cudnnDataTypeConvert(op->getDType());

        // get inputs
        checkCudnnError(cudnnCreateTensorDescriptor(&inputDesc));
        checkCudnnError(cudnnSetTensorNdDescriptor(
            inputDesc, cudnnDataType, dim.size(), dim.data(), stride.data()));

        // get outputs
        checkCudnnError(cudnnCreateTensorDescriptor(&outputDesc));
        checkCudnnError(cudnnSetTensorNdDescriptor(
            outputDesc, cudnnDataType, dim.size(), dim.data(), stride.data()));

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

        auto cudnnDataType = cudnnDataTypeConvert(op->getDType());

        // get inputs
        checkCudnnError(cudnnCreateTensorDescriptor(&inputDesc));
        checkCudnnError(cudnnSetTensor4dDescriptor(
            inputDesc, CUDNN_TENSOR_NCHW, cudnnDataType, dim_array[0],
            dim_array[1], dim_array[2], dim_array[3]));

        // get outputs
        checkCudnnError(cudnnCreateTensorDescriptor(&outputDesc));
        checkCudnnError(cudnnSetTensor4dDescriptor(
            outputDesc, CUDNN_TENSOR_NCHW, cudnnDataType, dim_array[0],
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

class LeakyReluCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<LeakyReluObj>(_op);
        void *const inputData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const outputData = (op->getOutput()->getRawDataPtr<void *>());
        auto alphaValue = op->getAlpha();
        size_t size = op->getOutput()->size();
        if (op->getDType() == DataType::Float32) {
            leaky_relu_kernel<float>((float *)inputData, (float *)outputData,
                                     size, alphaValue);
        } else {
            IT_TODO_HALT();
        }
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

REGISTER_KERNEL(Device::CUDA, OpType::Relu, ReluCudnn, "Relu_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Sigmoid, SigmoidCudnn, "Sigmoid_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Elu, EluCuda, "Elu_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::HardSigmoid, UnaryCuda,
                "Hard_Sigmoid_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::HardSwish, UnaryCuda, "Hard_Swish_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Tanh, TanhCudnn, "Tanh_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Abs, UnaryCuda, "Abs_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Sqrt, UnaryCuda, "Sqrt_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Gelu, UnaryCuda, "Gelu_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Silu, UnaryCuda, "Silu_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Neg, UnaryCuda, "Neg_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Erf, UnaryCuda, "Erf_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::LeakyRelu, LeakyReluCuda,
                "LeakyRelu_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Cast, CastCuda, "Cast_CUDA");

// REGISTER_KERNEL(Device::CUDA, OpType::Softmax, UnaryCuda,
// "Softmax_CUDA"); REGISTER_KERNEL(Device::CUDA, OpType::Relu,
// UnaryCuda,
//                 "Relu_CUDA");
// REGISTER_KERNEL(Device::CUDA, OpType::Sigmoid, UnaryCuda,
//                 "Sigmoid_CUDA");
// REGISTER_KERNEL(Device::CUDA, OpType::Tanh, UnaryCuda,
//                 "Tanh_CUDA");
// REGISTER_KERNEL(Device::CUDA, OpType::Abs, UnaryCuda,
//                 "Abs_CUDA");
}; // namespace infini

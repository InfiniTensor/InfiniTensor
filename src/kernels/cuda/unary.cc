#include "operators/unary.h"
#include "core/kernel.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_unary.h"

namespace infini {

class UnaryCuda : public Kernel {
    void compute(const Operator &_op, const PerfRecord &record,
                 const RuntimeObj *_context) const override {
        unary_kernel(_op);
    }

    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        compute(_op, {}, _context);
    }
    // Premise: op is idempotent since it is called multiple times.
    PerfRecord tune(const Operator &_op,
                    const RuntimeObj *_context) const override {
        PerfRecord ret;
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);
        ret.time = timeit([&]() { compute(_op, _context); },
                          [&]() { context->sync(); });
        return ret;
    }
};

class ActivationCudnn : public Kernel {
    virtual cudnnActivationMode_t getOpType() const = 0;
    virtual tuple<float, float> getAlphBeta() const { return {1.f, 0.f}; }
    void compute(const Operator &_op, const PerfRecord &record,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);

        void *const inputData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const outputData = (op->getOutput()->getRawDataPtr<void *>());

        cudnnTensorDescriptor_t inputDesc, outputDesc;
        auto dim = op->getInputs(0)->getDims();
        if (dim.size() != 4)
            IT_TODO_HALT();
        int n = dim[0], c = dim[1], h = dim[2], w = dim[3];

        // get inputs
        checkCudnnError(cudnnCreateTensorDescriptor(&inputDesc));
        checkCudnnError(cudnnSetTensor4dDescriptor(
            inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));

        // get outputs
        checkCudnnError(cudnnCreateTensorDescriptor(&outputDesc));
        checkCudnnError(cudnnSetTensor4dDescriptor(
            outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));

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

    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        compute(_op, {}, _context);
    }
    // Premise: op is idempotent since it is called multiple times.
    PerfRecord tune(const Operator &_op,
                    const RuntimeObj *_context) const override {
        PerfRecord ret;
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);
        ret.time = timeit([&]() { compute(_op, _context); },
                          [&]() { context->sync(); });
        return ret;
    }
};

class SoftmaxCudnn : public Kernel {
    virtual cudnnSoftmaxAlgorithm_t getAlgorithmType() const = 0;
    virtual cudnnSoftmaxMode_t getModeType() const = 0;
    virtual tuple<float, float> getAlphBeta() const { return {1.f, 0.f}; }
    void compute(const Operator &_op, const PerfRecord &record,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);

        void *const inputData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const outputData = (op->getOutput()->getRawDataPtr<void *>());

        cudnnTensorDescriptor_t inputDesc, outputDesc;
        auto dim = op->getInputs(0)->getDims();
        if (dim.size() != 4)
            IT_TODO_HALT();
        int n = dim[0], c = dim[1], h = dim[2], w = dim[3];

        // get inputs
        checkCudnnError(cudnnCreateTensorDescriptor(&inputDesc));
        checkCudnnError(cudnnSetTensor4dDescriptor(
            inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));

        // get outputs
        checkCudnnError(cudnnCreateTensorDescriptor(&outputDesc));
        checkCudnnError(cudnnSetTensor4dDescriptor(
            outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));

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

    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        compute(_op, {}, _context);
    }
    // Premise: op is idempotent since it is called multiple times.
    PerfRecord tune(const Operator &_op,
                    const RuntimeObj *_context) const override {
        PerfRecord ret;
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);
        ret.time = timeit([&]() { compute(_op, _context); },
                          [&]() { context->sync(); });
        return ret;
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

class NormalSoftmaxCudnn : public SoftmaxCudnn {
    cudnnSoftmaxAlgorithm_t getAlgorithmType() const override {
        return CUDNN_SOFTMAX_ACCURATE;
    }
    cudnnSoftmaxMode_t getModeType() const override {
        return CUDNN_SOFTMAX_MODE_INSTANCE;
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Softmax, DataType::Float32,
                NormalSoftmaxCudnn, "Softmax_CUDA_Float32");
REGISTER_KERNEL(Device::CUDA, OpType::Relu, DataType::Float32, ReluCudnn,
                "Relu_CUDA_Float32");
REGISTER_KERNEL(Device::CUDA, OpType::Sigmoid, DataType::Float32, SigmoidCudnn,
                "Sigmoid_CUDA_Float32");
REGISTER_KERNEL(Device::CUDA, OpType::Tanh, DataType::Float32, TanhCudnn,
                "Tanh_CUDA_Float32");
REGISTER_KERNEL(Device::CUDA, OpType::Abs, DataType::Float32, UnaryCuda,
                "Abs_CUDA_Float32");

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

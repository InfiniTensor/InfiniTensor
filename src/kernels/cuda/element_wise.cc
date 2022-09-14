#include "operators/element_wise.h"
#include "cuda/cuda_element_wise.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"

namespace infini {
class ElementWiseCudnn : public CudaKernelWithoutConfig {
    virtual cudnnOpTensorOp_t getOpType() const = 0;
    virtual tuple<float, float, float> getAlphBeta() const {
        return {1.f, 1.f, 0.f};
    }
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        cudnnTensorDescriptor_t aDesc, bDesc, cDesc;
        auto dim = op->getInputs(0)->getDims();
        if (dim.size() != 4)
            IT_TODO_HALT();
        int n = dim[0], c = dim[1], h = dim[2], w = dim[3];

        // get inputs
        checkCudnnError(cudnnCreateTensorDescriptor(&aDesc));
        checkCudnnError(cudnnSetTensor4dDescriptor(
            aDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));

        checkCudnnError(cudnnCreateTensorDescriptor(&bDesc));
        checkCudnnError(cudnnSetTensor4dDescriptor(
            bDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));

        // get outputs
        checkCudnnError(cudnnCreateTensorDescriptor(&cDesc));
        checkCudnnError(cudnnSetTensor4dDescriptor(
            cDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));

        // get op descriptor
        cudnnOpTensorDescriptor_t opDesc;
        checkCudnnError(cudnnCreateOpTensorDescriptor(&opDesc));
        checkCudnnError(cudnnSetOpTensorDescriptor(
            opDesc, getOpType(), CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));

        auto [aAlpha, bAlpha, beta] = getAlphBeta();
        cudnnStatus_t stat =
            cudnnOpTensor(context->cudnnHandle(), opDesc, &aAlpha, aDesc, aData,
                          &bAlpha, bDesc, bData, &beta, cDesc, cData);
        if (stat != CUDNN_STATUS_SUCCESS)
            return;

        // Destories in CUDA does not require sync. But cuDNN does not state
        // whether sync is required before destories.
        checkCudnnError(cudnnDestroyTensorDescriptor(aDesc));
        checkCudnnError(cudnnDestroyTensorDescriptor(bDesc));
        checkCudnnError(cudnnDestroyTensorDescriptor(cDesc));
        checkCudnnError(cudnnDestroyOpTensorDescriptor(opDesc));
    }
};

class AddCudnn : public ElementWiseCudnn {
    cudnnOpTensorOp_t getOpType() const override { return CUDNN_OP_TENSOR_ADD; }
};

class SubCudnn : public ElementWiseCudnn {
    cudnnOpTensorOp_t getOpType() const override { return CUDNN_OP_TENSOR_ADD; }
    tuple<float, float, float> getAlphBeta() const override {
        return {1.f, -1.f, 0.f};
    }
};

class MulCudnn : public ElementWiseCudnn {
    cudnnOpTensorOp_t getOpType() const override { return CUDNN_OP_TENSOR_MUL; }
};

class ElementWiseCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        float *const aData = (op->getInputs(0)->getRawDataPtr<float *>());
        float *const bData = (op->getInputs(1)->getRawDataPtr<float *>());
        float *const cData = (op->getOutput()->getRawDataPtr<float *>());

        auto dim = op->getInputs(0)->getDims();
        int n = dim[0], c = dim[1], h = dim[2], w = dim[3];
        if (op->getOpType() == OpType::Div)
            div_kernel(aData, bData, cData, n * c * h * w);
        else if (op->getOpType() == OpType::Pow)
            pow_kernel(aData, bData, cData, n * c * h * w);
        else
            IT_TODO_HALT();
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Add, DataType::Float32, AddCudnn,
                "Add_cuDNN_CUDA_Float32");
REGISTER_KERNEL(Device::CUDA, OpType::Sub, DataType::Float32, SubCudnn,
                "Sub_cuDNN_CUDA_Float32");
REGISTER_KERNEL(Device::CUDA, OpType::Mul, DataType::Float32, MulCudnn,
                "Mul_cuDNN_CUDA_Float32");

REGISTER_KERNEL(Device::CUDA, OpType::Div, DataType::Float32, ElementWiseCuda,
                "Div_CUDA_Float32");
REGISTER_KERNEL(Device::CUDA, OpType::Pow, DataType::Float32, ElementWiseCuda,
                "Pow__CUDA_Float32");
}; // namespace infini
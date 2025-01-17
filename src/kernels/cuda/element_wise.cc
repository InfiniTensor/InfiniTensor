#include "operators/element_wise.h"
#include "cuda/cuda_element_wise.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"

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

        auto aTensor = op->getInputs(0);
        auto bTensor = op->getInputs(1);
        auto cTensor = op->getOutput();

        // cudnnOpTensor only allows B to be broadcasted.
        if (aTensor->getDims() != cTensor->getDims()) {
            swap(aTensor, bTensor);
        }
        IT_ASSERT(aTensor->getDims() == cTensor->getDims(),
                  "Shape does not match.");

        void *const aData = (aTensor->getRawDataPtr<void *>());
        void *const bData = (bTensor->getRawDataPtr<void *>());
        void *const cData = (cTensor->getRawDataPtr<void *>());

        cudnnTensorDescriptor_t aDesc, bDesc, cDesc;
        auto a_dim = aTensor->getDims();
        auto b_dim = bTensor->getDims();
        auto c_dim = cTensor->getDims();

        if (a_dim.size() > 4 || b_dim.size() > 4 || c_dim.size() > 4)
            IT_TODO_HALT();

        int a[4] = {1, 1, 1, 1};
        int b[4] = {1, 1, 1, 1};
        int c[4] = {1, 1, 1, 1};

        std::copy(a_dim.begin(), a_dim.end(), a + (4 - a_dim.size()));
        std::copy(b_dim.begin(), b_dim.end(), b + (4 - b_dim.size()));
        std::copy(c_dim.begin(), c_dim.end(), c + (4 - c_dim.size()));

        auto cudnnDataType = cudnnDataTypeConvert(op->getDType());
        // get inputs
        checkCudnnError(cudnnCreateTensorDescriptor(&aDesc));
        checkCudnnError(cudnnSetTensor4dDescriptor(
            aDesc, CUDNN_TENSOR_NCHW, cudnnDataType, a[0], a[1], a[2], a[3]));

        checkCudnnError(cudnnCreateTensorDescriptor(&bDesc));
        checkCudnnError(cudnnSetTensor4dDescriptor(
            bDesc, CUDNN_TENSOR_NCHW, cudnnDataType, b[0], b[1], b[2], b[3]));

        // get outputs
        checkCudnnError(cudnnCreateTensorDescriptor(&cDesc));
        checkCudnnError(cudnnSetTensor4dDescriptor(
            cDesc, CUDNN_TENSOR_NCHW, cudnnDataType, c[0], c[1], c[2], c[3]));

        // get op descriptor
        cudnnOpTensorDescriptor_t opDesc;
        checkCudnnError(cudnnCreateOpTensorDescriptor(&opDesc));
        checkCudnnError(cudnnSetOpTensorDescriptor(
            opDesc, getOpType(), CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));

        auto [aAlpha, bAlpha, beta] = getAlphBeta();

        checkCudnnError(cudnnOpTensor(context->cudnnHandle(), opDesc, &aAlpha,
                                      aDesc, aData, &bAlpha, bDesc, bData,
                                      &beta, cDesc, cData));

        // Destories in CUDA does not require sync. But cuDNN does not state
        // whether sync is required before destories.
        checkCudnnError(cudnnDestroyTensorDescriptor(aDesc));
        checkCudnnError(cudnnDestroyTensorDescriptor(bDesc));
        checkCudnnError(cudnnDestroyTensorDescriptor(cDesc));
        checkCudnnError(cudnnDestroyOpTensorDescriptor(opDesc));
    }

    void compute(const Operator &op, const PerfRecord &record,
                 const RuntimeObj *context) const override {
        compute(op, context);
    }

  public:
    ElementWiseCudnn() {
        ComputeFuncPtr computePtr = [this](const Operator &op,
                                           const PerfRecord &record,
                                           const RuntimeObj *context) {
            this->compute(op, record, context);
        };
        funcVec.emplace_back(computePtr);
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

class MinCudnn : public ElementWiseCudnn {
    cudnnOpTensorOp_t getOpType() const override { return CUDNN_OP_TENSOR_MIN; }
};

class MaxCudnn : public ElementWiseCudnn {
    cudnnOpTensorOp_t getOpType() const override { return CUDNN_OP_TENSOR_MAX; }
};

class ElementWiseCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto a_dim = op->getInputs(0)->getDims();
        auto b_dim = op->getInputs(1)->getDims();
        auto c_dim = op->getOutput()->getDims();
        const int dType = _op->getDType().getIndex();

        // Use optimized kernel if b is constant
        if (b_dim.size() == 0) {
            if (op->getOpType() == OpType::Div) {
                div_const_kernel(dType, aData, bData, cData,
                                 op->getOutput()->size());
                return;
            } else if (op->getOpType() == OpType::Pow) {
                pow_const_kernel(dType, aData, bData, cData,
                                 op->getOutput()->size());
                return;
            }
        }

        if (a_dim.size() > 4 || b_dim.size() > 4 || c_dim.size() > 4)
            IT_TODO_HALT();

        int a[4] = {1, 1, 1, 1};
        int b[4] = {1, 1, 1, 1};
        int c[4] = {1, 1, 1, 1};

        std::copy(a_dim.begin(), a_dim.end(), a + (4 - a_dim.size()));
        std::copy(b_dim.begin(), b_dim.end(), b + (4 - b_dim.size()));
        std::copy(c_dim.begin(), c_dim.end(), c + (4 - c_dim.size()));

        if (op->getOpType() == OpType::Div) {
            div_kernel(dType, aData, bData, cData, a[0], a[1], a[2], a[3], b[0],
                       b[1], b[2], b[3], c[0], c[1], c[2], c[3]);
        } else if (op->getOpType() == OpType::Add) {
            add_kernel(dType, aData, bData, cData, a[0], a[1], a[2], a[3], b[0],
                       b[1], b[2], b[3], c[0], c[1], c[2], c[3]);
        } else if (op->getOpType() == OpType::Pow) {
            pow_kernel(dType, aData, bData, cData, a[0], a[1], a[2], a[3], b[0],
                       b[1], b[2], b[3], c[0], c[1], c[2], c[3]);
        } else if (op->getOpType() == OpType::Less) {
            less_kernel(dType, aData, bData, cData, a[0], a[1], a[2], a[3],
                        b[0], b[1], b[2], b[3], c[0], c[1], c[2], c[3]);
        } else {
            IT_TODO_HALT();
        }
    }
};

// REGISTER_KERNEL(Device::CUDA, OpType::Add, AddCudnn, "Add_cuDNN_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Sub, SubCudnn, "Sub_cuDNN_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Mul, MulCudnn, "Mul_cuDNN_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Min, MinCudnn, "Min_cuDNN_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Max, MaxCudnn, "Max_cuDNN_CUDA");

REGISTER_KERNEL(Device::CUDA, OpType::Div, ElementWiseCuda, "Div_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Pow, ElementWiseCuda, "Pow_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Less, ElementWiseCuda, "Less_CUDA");

}; // namespace infini

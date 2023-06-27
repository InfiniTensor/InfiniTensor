#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"
#include "operators/unary.h"

namespace infini {
class UnaryCnnl : public BangKernelWithoutConfig {
    virtual cnnlActivationMode_t getOpType() const = 0;
    virtual float getCoef() const = 0;
    virtual tuple<float, float> getAlphBeta() const { return {1.f, 0.f}; }
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        cnnlTensorDescriptor_t aDesc, cDesc;
        auto aDim = op->getInputs(0)->getDims();
        auto cDim = op->getOutput()->getDims();

        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, aDim.size(),
                                               aDim.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, cDim.size(),
                                               cDim.data()));
        cnnlActivationDescriptor_t opDesc;
        checkCnnlError(cnnlCreateActivationDescriptor(&opDesc));
        checkCnnlError(cnnlSetActivationDescriptor(
            opDesc, getOpType(), CNNL_NOT_PROPAGATE_NAN, getCoef()));

        auto [alpha, beta] = getAlphBeta();
        cnnlStatus_t stat =
            cnnlActivationForward(context->cnnlHandle(), opDesc, &alpha, aDesc,
                                  aData, &beta, cDesc, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;
        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
        checkCnnlError(cnnlDestroyActivationDescriptor(opDesc));
    }
};

class RoundCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        cnnlTensorDescriptor_t aDesc, cDesc;
        auto aDim = op->getInputs(0)->getDims();
        auto cDim = op->getOutput()->getDims();

        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, aDim.size(),
                                               aDim.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, cDim.size(),
                                               cDim.data()));
        cnnlStatus_t stat =
            cnnlRound(context->cnnlHandle(), aDesc, aData, cDesc, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;
        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
    }
};

class SquareCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        cnnlTensorDescriptor_t aDesc, cDesc;
        auto aDim = op->getInputs(0)->getDims();
        auto cDim = op->getOutput()->getDims();

        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, aDim.size(),
                                               aDim.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, cDim.size(),
                                               cDim.data()));
        cnnlStatus_t stat =
            cnnlSquare(context->cnnlHandle(), aDesc, aData, cDesc, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;
        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
    }
};

class PReluCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<PReluObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        cnnlTensorDescriptor_t aDesc, bDesc, cDesc;
        auto aDim = op->getInputs(0)->getDims();
        auto bDim = op->getInputs(1)->getDims();
        auto cDim = op->getOutput()->getDims();

        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, aDim.size(),
                                               aDim.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&bDesc));
        checkCnnlError(cnnlSetTensorDescriptor(bDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, bDim.size(),
                                               bDim.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, cDim.size(),
                                               cDim.data()));

        cnnlStatus_t stat = cnnlPrelu(context->cnnlHandle(), aDesc, aData,
                                      bDesc, bData, cDesc, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(bDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
    }
};

class ReluCnnl : public UnaryCnnl {
    cnnlActivationMode_t getOpType() const override {
        return CNNL_ACTIVATION_RELU;
    }
    float getCoef() const override { return 0.0; }
};

class SigmoidCnnl : public UnaryCnnl {
    cnnlActivationMode_t getOpType() const override {
        return CNNL_ACTIVATION_SIGMOID;
    }
    float getCoef() const override { return 0.0; }
};

class TanhCnnl : public UnaryCnnl {
    cnnlActivationMode_t getOpType() const override {
        return CNNL_ACTIVATION_TANH;
    }
    float getCoef() const override { return 0.0; }
};

REGISTER_KERNEL(Device::BANG, OpType::Relu, DataType::Float32, ReluCnnl,
                "Relu_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::PRelu, DataType::Float32, PReluCnnl,
                "PRelu_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::Sigmoid, DataType::Float32, SigmoidCnnl,
                "Sigmoid_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::Tanh, DataType::Float32, TanhCnnl,
                "Tanh_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::Round, DataType::Float32, RoundCnnl,
                "Round_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::Square, DataType::Float32, SquareCnnl,
                "Square_cnnl_BANG_Float32");

}; // namespace infini

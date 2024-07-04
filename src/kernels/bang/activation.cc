#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"
#include "bang/bang_softmax.h"
#include "operators/softmax.h"
#include "operators/unary.h"
#include <iostream>

namespace infini {
class UnaryCnnl : public BangKernelWithoutConfig {
    virtual cnnlActivationMode_t getOpType() const = 0;
    virtual float getCoef() const = 0;
    virtual tuple<float, float> getAlphBeta() const { return {1.f, 0.f}; }
    virtual float getSlicedDim() const { return 0.0; }
    virtual float getGamma() const { return 0.0; }
    virtual float getScale() const { return 0.0; }

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
        checkCnnlError(cnnlSetTensorDescriptor(
            aDesc, CNNL_LAYOUT_NCHW, cnnlDataTypeConvert(op->getDType()),
            aDim.size(), aDim.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            cDesc, CNNL_LAYOUT_NCHW, cnnlDataTypeConvert(op->getDType()),
            cDim.size(), cDim.data()));
        cnnlActivationDescriptor_t opDesc;
        checkCnnlError(cnnlCreateActivationDescriptor(&opDesc));
        checkCnnlError(cnnlSetActivationDescriptor_v5(
            opDesc, getOpType(), CNNL_ACTIVATION_HIGH_PRECISION,
            CNNL_NOT_PROPAGATE_NAN, getCoef(), getSlicedDim(), getGamma(),
            getScale(), true));

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
        checkCnnlError(cnnlSetTensorDescriptor(
            aDesc, CNNL_LAYOUT_NCHW, cnnlDataTypeConvert(op->getDType()),
            aDim.size(), aDim.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            cDesc, CNNL_LAYOUT_NCHW, cnnlDataTypeConvert(op->getDType()),
            cDim.size(), cDim.data()));
        cnnlStatus_t stat =
            cnnlRound(context->cnnlHandle(), aDesc, aData, cDesc, cData);
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

        if (auto alignSize = aDim.size() - bDim.size(); alignSize) {
            bDim.insert(bDim.begin(), alignSize, 1);
        }

        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            aDesc, CNNL_LAYOUT_NCHW, cnnlDataTypeConvert(op->getDType()),
            aDim.size(), aDim.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&bDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            bDesc, CNNL_LAYOUT_NCHW, cnnlDataTypeConvert(op->getDType()),
            bDim.size(), bDim.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            cDesc, CNNL_LAYOUT_NCHW, cnnlDataTypeConvert(op->getDType()),
            cDim.size(), cDim.data()));

        cnnlStatus_t stat = cnnlPrelu(context->cnnlHandle(), aDesc, aData,
                                      bDesc, bData, cDesc, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(bDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
    }
};

class SoftmaxCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<SoftmaxObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        cnnlTensorDescriptor_t aDesc, cDesc;
        auto aDim = op->getInputs(0)->getDims();
        int axis = op->getAxis();
        int nDim = aDim.size();
        if(axis == 0 || axis == nDim - 1){
            int stride = 1;
            int dimsize = aDim[axis];
            int num = 1;
            int othersize = 1;
            int frontsize = 1;

            for (int s = nDim - 1; s >= 0; s--) {
                num *= aDim[s];
                if (s > axis) {
                    stride *= aDim[s];
                }
                if (s < axis) {
                    frontsize *= aDim[s];
                }
                if (s != axis) {
                    othersize *= aDim[s];
                }
            }
            softmaxKernel(context->cnnlHandle(), (float *)cData,
                          (float *)aData, othersize, dimsize, frontsize,
                          stride, axis, nDim);
        }
        else{
            cnnlSoftmaxMode_t mode;
        
            std::vector<int> inDim = {1, 1, 1};
            std::vector<int> outDim = inDim;

            if (nDim >= 3) {
                if (axis == 0) {
                    mode = CNNL_SOFTMAX_MODE_HIGH_DIMENSION;
                    inDim[0] = aDim[0];
                    inDim[1] = aDim[1];
                    for (int i = 2; i < nDim; ++i) {
                        inDim[2] *= aDim[i];
                    }
                    outDim = inDim;
                } else if (axis == nDim - 1) {
                    mode = CNNL_SOFTMAX_MODE_LOW_DIMENSION;
                    inDim[0] = aDim[0];
                    for (int i = 1; i < axis; ++i) {
                        inDim[1] *= aDim[i];
                    }
                    inDim[2] = aDim[axis];
                    outDim = inDim;
                } else {
                    mode = CNNL_SOFTMAX_MODE_MEDIUM_DIMENSION;
                    for (int i = 0; i < axis; ++i) {
                        inDim[0] *= aDim[i];
                    }
                    inDim[1] = aDim[axis];
                    for (int i = axis + 1; i < nDim; ++i) {
                        inDim[2] *= aDim[i];
                    }
                    outDim = inDim;
                }
            } else if (nDim == 2) {
                if (axis == 0) {
                    mode = CNNL_SOFTMAX_MODE_HIGH_DIMENSION;
                    inDim = aDim;
                    inDim.push_back(1);
                    outDim = inDim;
                } else {
                    mode = CNNL_SOFTMAX_MODE_LOW_DIMENSION;
                    inDim = aDim;
                    inDim.insert(inDim.begin(), 1);
                    outDim = inDim;
                }
            } else {
                mode = CNNL_SOFTMAX_MODE_HIGH_DIMENSION;
                inDim = aDim;
                inDim.push_back(1);
                inDim.push_back(1);
                outDim = inDim;
            }

            checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
            checkCnnlError(cnnlSetTensorDescriptor(
                aDesc, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(op->getDType()),
                inDim.size(), inDim.data()));
            checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
            checkCnnlError(cnnlSetTensorDescriptor(
                cDesc, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(op->getDType()),
                outDim.size(), outDim.data()));
            float alpha = 1.0;
            float beta = 0.0;
            cnnlStatus_t stat =
                cnnlSoftmaxForward_v2(context->cnnlHandle(), CNNL_SOFTMAX_ACCURATE,
                                    mode, CNNL_COMPUTATION_ULTRAHIGH_PRECISION,
                                    &alpha, aDesc, aData, &beta, cDesc, cData);
            if (stat != CNNL_STATUS_SUCCESS)
                return;
            checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
            checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
        }
        
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

class HardSwishCnnl : public UnaryCnnl {
    cnnlActivationMode_t getOpType() const override {
        return CNNL_ACTIVATION_HARDSWISH;
    }
    float getCoef() const override { return 0.0; }
};

class HardSigmoidCnnl : public UnaryCnnl {
    cnnlActivationMode_t getOpType() const override {
        return CNNL_ACTIVATION_HARDSIGMOID;
    }
    float getCoef() const override { return 0.0; }
    float getGamma() const override { return 1.f / 6.f; }
    float getScale() const override { return 0.5f; }
};

class LeakyReluCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<LeakyReluObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        cnnlTensorDescriptor_t aDesc, cDesc;
        auto aDim = op->getInputs(0)->getDims();
        auto cDim = op->getOutput()->getDims();
        auto coef = op->getAlpha();

        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            aDesc, CNNL_LAYOUT_NCHW, cnnlDataTypeConvert(op->getDType()),
            aDim.size(), aDim.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            cDesc, CNNL_LAYOUT_NCHW, cnnlDataTypeConvert(op->getDType()),
            cDim.size(), cDim.data()));
        cnnlActivationDescriptor_t opDesc;
        checkCnnlError(cnnlCreateActivationDescriptor(&opDesc));
        checkCnnlError(cnnlSetActivationDescriptor_v5(
            opDesc, CNNL_ACTIVATION_LEAKYRELU, CNNL_ACTIVATION_HIGH_PRECISION,
            CNNL_NOT_PROPAGATE_NAN, coef, 0.0, 0.0, 0.0, true));

        float alpha = 1.f, beta = 0.f;
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

REGISTER_KERNEL(Device::BANG, OpType::Relu, ReluCnnl, "Relu_cnnl_BANG");
REGISTER_KERNEL(Device::BANG, OpType::PRelu, PReluCnnl, "PRelu_cnnl_BANG");
REGISTER_KERNEL(Device::BANG, OpType::LeakyRelu, LeakyReluCnnl,
                "LeakyRelu_cnnl_BANG");
REGISTER_KERNEL(Device::BANG, OpType::Sigmoid, SigmoidCnnl,
                "Sigmoid_cnnl_BANG");
REGISTER_KERNEL(Device::BANG, OpType::Round, RoundCnnl, "Round_cnnl_BANG");
REGISTER_KERNEL(Device::BANG, OpType::Softmax, SoftmaxCnnl,
                "Softmax_cnnl_BANG");
REGISTER_KERNEL(Device::BANG, OpType::HardSigmoid, HardSigmoidCnnl,
                "HardSigmoid_cnnl_BANG");
REGISTER_KERNEL(Device::BANG, OpType::HardSwish, HardSwishCnnl,
                "HardSwish_cnnl_BANG");

}; // namespace infini

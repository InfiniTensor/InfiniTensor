#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"
#include "operators/softmax.h"
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
        checkCnnlError(cnnlSetActivationDescriptor_v2(
            opDesc, getOpType(), CNNL_ACTIVATION_HIGH_PRECISION,
            CNNL_NOT_PROPAGATE_NAN, getCoef()));

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

class SoftmaxCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<SoftmaxObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        cnnlTensorDescriptor_t aDesc, cDesc;
        auto aDim = op->getInputs(0)->getDims();

        cnnlSoftmaxMode_t mode;
        size_t axis = op->getAxis();
        std::vector<int> inDim = {1, 1, 1};
        std::vector<int> outDim = inDim;

        if (aDim.size() >= 3) {
            if (axis == 0) {
                mode = CNNL_SOFTMAX_MODE_HIGH_DIMENSION;
                inDim[0] = aDim[0];
                inDim[1] = aDim[1];
                for (size_t i = 2; i < aDim.size(); ++i) {
                    inDim[2] *= aDim[i];
                }
                outDim = inDim;
            } else if (axis == aDim.size() - 1) {
                mode = CNNL_SOFTMAX_MODE_LOW_DIMENSION;
                inDim[0] = aDim[0];
                for (size_t i = 1; i < axis; ++i) {
                    inDim[1] *= aDim[i];
                }
                inDim[2] = aDim[axis];
                outDim = inDim;
            } else {
                mode = CNNL_SOFTMAX_MODE_MEDIUM_DIMENSION;
                for (size_t i = 0; i < axis; ++i) {
                    inDim[0] *= aDim[i];
                }
                inDim[1] = aDim[axis];
                for (size_t i = axis + 1; i < aDim.size(); ++i) {
                    inDim[2] *= aDim[i];
                }
                outDim = inDim;
            }
        } else if (aDim.size() == 2) {
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
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_ARRAY,
                                               CNNL_DTYPE_FLOAT, inDim.size(),
                                               inDim.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_ARRAY,
                                               CNNL_DTYPE_FLOAT, outDim.size(),
                                               outDim.data()));
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

REGISTER_KERNEL(Device::BANG, OpType::Relu, DataType::Float32, ReluCnnl,
                "Relu_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::PRelu, DataType::Float32, PReluCnnl,
                "PRelu_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::Sigmoid, DataType::Float32, SigmoidCnnl,
                "Sigmoid_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::Round, DataType::Float32, RoundCnnl,
                "Round_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::Softmax, DataType::Float32, SoftmaxCnnl,
                "Softmax_cnnl_BANG_Float32");

}; // namespace infini

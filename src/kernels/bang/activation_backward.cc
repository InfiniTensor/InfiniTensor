#include "operators/activation_backward.h"
#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"

namespace infini {
class ActivationBackwardCnnl : public BangKernelWithoutConfig {
    virtual cnnlActivationMode_t getOpType() const = 0;
    virtual float getCoef() const = 0;
    virtual tuple<float, float> getAlphBeta() const { return {1.f, 0.f}; }
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ActivationBackwardObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const yData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const diffYData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const xData = (op->getInputs(2)->getRawDataPtr<void *>());
        void *const diffXData = (op->getOutput()->getRawDataPtr<void *>());

        cnnlTensorDescriptor_t yDesc, diffYDesc, xDesc, diffXDesc;
        auto dim = op->getInputs(0)->getDims();
        if (dim.size() != 4)
            IT_TODO_HALT();

        int dim_array[4] = {dim[0], dim[1], dim[2], dim[3]};
        // get inputs
        checkCnnlError(cnnlCreateTensorDescriptor(&yDesc));
        checkCnnlError(cnnlSetTensorDescriptor(yDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));
        // get inputs
        checkCnnlError(cnnlCreateTensorDescriptor(&diffYDesc));
        checkCnnlError(cnnlSetTensorDescriptor(diffYDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));
        // get inputs
        checkCnnlError(cnnlCreateTensorDescriptor(&xDesc));
        checkCnnlError(cnnlSetTensorDescriptor(xDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));
        // get outputs
        checkCnnlError(cnnlCreateTensorDescriptor(&diffXDesc));
        checkCnnlError(cnnlSetTensorDescriptor(diffXDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));

        // get op descriptor
        cnnlActivationDescriptor_t opDesc;
        checkCnnlError(cnnlCreateActivationDescriptor(&opDesc));
        checkCnnlError(cnnlSetActivationDescriptor(
            opDesc, getOpType(), CNNL_NOT_PROPAGATE_NAN, getCoef()));

        auto [alpha, beta] = getAlphBeta();
        cnnlStatus_t stat = cnnlActivationBackward(
            context->cnnlHandle(), opDesc, &alpha, yDesc, yData, diffYDesc,
            diffYData, xDesc, xData, &beta, diffXDesc, diffXData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        // Destories in BANG does not require sync. But cnnl does not state
        // whether sync is required before destories.
        checkCnnlError(cnnlDestroyTensorDescriptor(yDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(diffYDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(xDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(diffXDesc));
        checkCnnlError(cnnlDestroyActivationDescriptor(opDesc));
    }
};

class ReluBackwardCnnl : public ActivationBackwardCnnl {
    cnnlActivationMode_t getOpType() const override {
        return CNNL_ACTIVATION_RELU;
    }
    float getCoef() const override { return 0.0; }
};

class SigmoidBackwardCnnl : public ActivationBackwardCnnl {
    cnnlActivationMode_t getOpType() const override {
        return CNNL_ACTIVATION_SIGMOID;
    }
    float getCoef() const override { return 0.0; }
};

class TanhBackwardCnnl : public ActivationBackwardCnnl {
    cnnlActivationMode_t getOpType() const override {
        return CNNL_ACTIVATION_TANH;
    }
    float getCoef() const override { return 0.0; }
};

REGISTER_KERNEL(Device::BANG, OpType::ReluBackward, DataType::Float32,
                ReluBackwardCnnl, "ReluBackward_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::SigmoidBackward, DataType::Float32,
                SigmoidBackwardCnnl, "SigmoidBackward_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::TanhBackward, DataType::Float32,
                TanhBackwardCnnl, "TanhBackward_cnnl_BANG_Float32");

}; // namespace infini

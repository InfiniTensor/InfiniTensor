#include "operators/pooling.h"
#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"

namespace infini {
class PoolingCnnl : public BangKernelWithoutConfig {
    virtual cnnlPoolingMode_t getPoolingMode() const = 0;
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<PoolingObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);
        void *const inData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const outData = (op->getOutput()->getRawDataPtr<void *>());

        const auto [n, c, h, w, kh, kw] = op->getNCHWRS();
        const auto [ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();

        // get inputs
        int inArray[4] = {n, c, h, w};
        cnnlTensorDescriptor_t inDesc;
        checkCnnlError(cnnlCreateTensorDescriptor(&inDesc));
        checkCnnlError(cnnlSetTensorDescriptor(inDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, inArray));

        // get maxpool descriptor
        cnnlPoolingDescriptor_t poolingDesc;
        checkCnnlError(cnnlCreatePoolingDescriptor(&poolingDesc));
        checkCnnlError(cnnlSetPooling2dDescriptor_v2(
            poolingDesc, getPoolingMode(), CNNL_NOT_PROPAGATE_NAN, kh, kw, ph,
            ph, pw, pw, sh, sw, dh, dw, false));

        // get outputs
        auto outVec = op->getOutput()->getDims();
        int outArray[4] = {outVec[0], outVec[1], outVec[2], outVec[3]};
        cnnlTensorDescriptor_t outDesc;
        checkCnnlError(cnnlCreateTensorDescriptor(&outDesc));
        checkCnnlError(cnnlSetTensorDescriptor(outDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, outArray));
        size_t wsSize;
        cnnlGetPoolingWorkspaceSize(context->cnnlHandle(), getPoolingMode(),
                                    outVec[3], outVec[2], &wsSize);
        BangPtr wsData = context->getWorkspace(wsSize);

        float alpha = 1.f, beta = 0.f;
        checkCnnlError(cnnlPoolingForward(context->cnnlHandle(), poolingDesc,
                                          &alpha, inDesc, inData, &beta,
                                          outDesc, outData, wsData, wsSize));

        // Destories in BANG does not require sync. But cnnl does not state
        // whether sync is required before destories.
        checkCnnlError(cnnlDestroyTensorDescriptor(inDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(outDesc));
        checkCnnlError(cnnlDestroyPoolingDescriptor(poolingDesc));
    }
};

class maxPoolCnnl : public PoolingCnnl {
    cnnlPoolingMode_t getPoolingMode() const override {
        return CNNL_POOLING_MAX;
    }
};

class avgPoolCnnl : public PoolingCnnl {
    cnnlPoolingMode_t getPoolingMode() const override {
        return CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    }
};

REGISTER_KERNEL(Device::BANG, OpType::MaxPool, DataType::Float32, maxPoolCnnl,
                "MaxPool_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::AvgPool, DataType::Float32, avgPoolCnnl,
                "AvgPool_cnnl_BANG_Float32");
}; // namespace infini

#include "operators/layer_norm.h"
#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"

namespace infini {

class LayerNormCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<LayerNormObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const inputData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const scaleData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *biasData = NULL;
        if (op->numInputs() == 3) {
            biasData = (op->getInputs(2)->getRawDataPtr<void *>());
        }
        void *const outputData = (op->getOutput()->getRawDataPtr<void *>());

        auto inDims = op->getInputs(0)->getDims();
        auto outDims = op->getOutput()->getDims();
        auto fiterDims = op->getOutput(1)->getDims();

        float eps = op->getEps();
        const int axis = op->getAxis();

        cnnlTensorDescriptor_t inDesc, fiterDesc, outDesc;

        checkCnnlError(cnnlCreateTensorDescriptor(&inDesc));
        checkCnnlError(cnnlSetTensorDescriptor(inDesc, CNNL_LAYOUT_ARRAY,
                                               CNNL_DTYPE_FLOAT, inDims.size(),
                                               inDims.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&fiterDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            fiterDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, fiterDims.size(),
            fiterDims.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&outDesc));
        checkCnnlError(cnnlSetTensorDescriptor(outDesc, CNNL_LAYOUT_ARRAY,
                                               CNNL_DTYPE_FLOAT, outDims.size(),
                                               outDims.data()));
        size_t wsSize;
        cnnlGetLayerNormOpWorkspaceSize(context->cnnlHandle(), axis, inDesc,
                                        &wsSize);
        BangPtr wsData = context->getWorkspace(wsSize);

        cnnlStatus_t stat = cnnlLayerNormForward(
            context->cnnlHandle(), inDesc, inputData, axis, fiterDesc,
            scaleData, biasData, eps, wsData, wsSize, outDesc, outputData,
            inDesc, NULL, NULL);

        if (stat != CNNL_STATUS_SUCCESS)
            return;

        checkCnnlError(cnnlDestroyTensorDescriptor(inDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(fiterDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(outDesc));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::LayerNormalization, LayerNormCnnl,
                "LayerNorm_BANG");

}; // namespace infini

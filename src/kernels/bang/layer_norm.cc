#include "operators/layer_norm.h"
#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"

namespace infini {

class LayerNormCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<LayerNormObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const inputData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const scaleData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *biasData = NULL;
        if (op->numInputs() == 3) {
            biasData = (op->getInputs(2)->getRawDataPtr<void *>());
        }
        void *const outputData = (op->getOutput()->getRawDataPtr<void *>());

        auto inDims = op->getInputs(0)->getDims();
        auto fiterDims = op->getInputs(1)->getDims();
        auto outDims = op->getOutput()->getDims();

        float eps = op->getEps();
        const int axis = op->getAxis();

        Shape outMeanDims(outDims);
        outMeanDims.erase(outMeanDims.begin() + axis);

        cnnlTensorDescriptor_t inDesc, fiterDesc, outDesc, outMeanDesc;
        checkCnnlError(cnnlCreateTensorDescriptor(&inDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            inDesc, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(op->getDType()),
            inDims.size(), inDims.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&fiterDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            fiterDesc, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(op->getDType()),
            fiterDims.size(), fiterDims.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&outDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            outDesc, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(op->getDType()),
            outDims.size(), outDims.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&outMeanDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            outMeanDesc, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(op->getDType()),
            outMeanDims.size(), outMeanDims.data()));
        size_t wsSize;
        cnnlGetLayerNormOpWorkspaceSize(context->cnnlHandle(), axis, inDesc,
                                        &wsSize);
        BangPtr wsData = context->getWorkspace(wsSize);
        size_t meanSize =
            cnnlGetTensorElementNum(outMeanDesc) * op->getDType().getSize();
        BangPtr meanData = context->getWorkspace(meanSize);
        BangPtr rstdData = context->getWorkspace(meanSize);

        cnnlStatus_t stat = cnnlLayerNormForward(
            context->cnnlHandle(), inDesc, inputData, axis, fiterDesc,
            scaleData, biasData, eps, wsData, wsSize, outDesc, outputData,
            outMeanDesc, meanData, rstdData);

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

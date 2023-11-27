#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"
#include "operators/batch_norm.h"

namespace infini {
class BatchNormCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<BatchNormObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const input = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const mean = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const var = (op->getInputs(2)->getRawDataPtr<void *>());
        void *const scale = (op->getInputs(3)->getRawDataPtr<void *>());
        void *const bias = (op->getInputs(4)->getRawDataPtr<void *>());
        void *const output = (op->getOutput()->getRawDataPtr<void *>());

        auto dims = op->getInputs(0)->getDims();
        if (dims.size() != 4)
            IT_TODO_HALT();

        int dimsTrans[4] = {dims[0], dims[2], dims[3], dims[1]};
        int permute[4] = {0, 2, 3, 1};
        int permuteOut[4] = {0, 3, 1, 2};

        // get inputs
        cnnlTensorDescriptor_t inDesc, intransDesc;
        checkCnnlError(cnnlCreateTensorDescriptor(&inDesc));
        checkCnnlError(cnnlCreateTensorDescriptor(&intransDesc));
        checkCnnlError(cnnlSetTensorDescriptor(inDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, dims.size(),
                                               dims.data()));
        checkCnnlError(cnnlSetTensorDescriptor(intransDesc, CNNL_LAYOUT_NHWC,
                                               CNNL_DTYPE_FLOAT, dims.size(),
                                               dimsTrans));
        cnnlTransposeDescriptor_t opDesc;
        checkCnnlError(cnnlCreateTransposeDescriptor(&opDesc));
        checkCnnlError(cnnlSetTransposeDescriptor(opDesc, 4, permute));
        size_t wsSize;
        cnnlGetTransposeWorkspaceSize(context->cnnlHandle(), inDesc, opDesc,
                                      &wsSize);
        BangPtr wsData = context->getWorkspace(wsSize);
        BangPtr inputTrans = context->getWorkspace(
            cnnlGetTensorElementNum(inDesc) * sizeof(float));
        BangPtr outputTrans = context->getWorkspace(
            cnnlGetTensorElementNum(inDesc) * sizeof(float));
        cnnlStatus_t stat =
            cnnlTranspose_v2(context->cnnlHandle(), opDesc, inDesc, input,
                             intransDesc, inputTrans, wsData, wsSize);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        // get bnScaleBiasMeanVarDesc
        auto dimsScaleBiasMeanVar = op->getInputs(1)->getDims();
        cnnlTensorDescriptor_t paraDesc;
        checkCnnlError(cnnlCreateTensorDescriptor(&paraDesc));
        checkCnnlError(cnnlSetTensorDescriptor(paraDesc, CNNL_LAYOUT_ARRAY,
                                               CNNL_DTYPE_FLOAT, 1,
                                               dimsScaleBiasMeanVar.data()));

        float alpha = 1.f, beta = 0.f;
        // This mode is intended for use after convolutional layers
        stat = cnnlBatchNormForwardInference(
            context->cnnlHandle(), &alpha, &beta, intransDesc, inputTrans,
            paraDesc, scale, bias, mean, var, op->getEps(), intransDesc,
            outputTrans);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        cnnlTransposeDescriptor_t op2Desc;
        checkCnnlError(cnnlCreateTransposeDescriptor(&op2Desc));
        checkCnnlError(cnnlSetTransposeDescriptor(op2Desc, 4, permuteOut));
        cnnlGetTransposeWorkspaceSize(context->cnnlHandle(), intransDesc,
                                      op2Desc, &wsSize);
        BangPtr ws2Data = context->getWorkspace(wsSize);
        stat = cnnlTranspose_v2(context->cnnlHandle(), op2Desc, intransDesc,
                                outputTrans, inDesc, output, ws2Data, wsSize);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        // Destories in BANG does not require sync. But cnnl does not state
        // whether sync is required before destories.
        checkCnnlError(cnnlDestroyTensorDescriptor(inDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(intransDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(paraDesc));
        checkCnnlError(cnnlDestroyTransposeDescriptor(opDesc));
        checkCnnlError(cnnlDestroyTransposeDescriptor(op2Desc));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::BatchNormalization, DataType::Float32,
                BatchNormCnnl, "BatchNorm_cnnl_BANG_Float32");

}; // namespace infini

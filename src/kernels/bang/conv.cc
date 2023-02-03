#include "operators/conv.h"
#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"

namespace infini {
class ConvCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ConvObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        const auto [ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
        const auto [n, c, h, w, f, r, s] = op->getNCHWFRS();
        const int cpg = op->getChannelPerGroup();
        const int g = c / cpg;

        int pad[4] = {ph, ph, pw, pw};
        int stride[2] = {sh, sw};
        int dilation[2] = {dh, dw};

        cnnlConvolutionDescriptor_t convDesc;
        checkCnnlError(cnnlCreateConvolutionDescriptor(&convDesc));
        checkCnnlError(cnnlSetConvolutionDescriptor(
            convDesc, 4, pad, stride, dilation, g, CNNL_DTYPE_FLOAT));

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        cnnlTensorDescriptor_t aInDesc, aDesc, bInDesc, bDesc, cInDesc, cDesc;
        auto dimInputs0 = op->getInputs(0)->getDims();
        auto dimInputs1 = op->getInputs(1)->getDims();
        auto dimOutput = op->getOutput()->getDims();

        if (dimInputs0.size() != 4)
            IT_TODO_HALT();
        if (dimInputs1.size() != 4)
            IT_TODO_HALT();
        if (dimOutput.size() != 4)
            IT_TODO_HALT();

        int inputs0[4] = {dimInputs0[0], dimInputs0[1], dimInputs0[2],
                          dimInputs0[3]};
        int inputs0Array[4] = {dimInputs0[0], dimInputs0[2], dimInputs0[3],
                               dimInputs0[1]};
        int inputs1[4] = {dimInputs1[0], dimInputs1[1], dimInputs1[2],
                          dimInputs1[3]};
        int inputs1Array[4] = {dimInputs1[0], dimInputs1[2], dimInputs1[3],
                               dimInputs1[1]};
        int output[4] = {dimOutput[0], dimOutput[1], dimOutput[2],
                         dimOutput[3]};
        int outputArray[4] = {dimOutput[0], dimOutput[2], dimOutput[3],
                              dimOutput[1]};

        // get inputs
        checkCnnlError(cnnlCreateTensorDescriptor(&aInDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aInDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, inputs0));

        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            aDesc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_FLOAT, 4, inputs0Array));

        checkCnnlError(cnnlCreateTensorDescriptor(&bInDesc));
        checkCnnlError(cnnlSetTensorDescriptor(bInDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, inputs1));

        checkCnnlError(cnnlCreateTensorDescriptor(&bDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            bDesc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_FLOAT, 4, inputs1Array));

        int permute[4] = {0, 2, 3, 1};
        cnnlTransposeDescriptor_t opDesc;
        checkCnnlError(cnnlCreateTransposeDescriptor(&opDesc));
        checkCnnlError(cnnlSetTransposeDescriptor(opDesc, 4, permute));

        size_t wsSize;
        cnnlGetTransposeWorkspaceSize(context->cnnlHandle(), aInDesc, opDesc,
                                      &wsSize);
        BangPtr wsData = context->getWorkspace(wsSize);
        BangPtr aDataOut = context->getWorkspace(
            cnnlGetTensorElementNum(aInDesc) * sizeof(float));
        cnnlStatus_t stat =
            cnnlTranspose_v2(context->cnnlHandle(), opDesc, aInDesc, aData,
                             aDesc, aDataOut, wsData, wsSize);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        cnnlGetTransposeWorkspaceSize(context->cnnlHandle(), bInDesc, opDesc,
                                      &wsSize);
        wsData = context->getWorkspace(wsSize);
        BangPtr bDataOut = context->getWorkspace(
            cnnlGetTensorElementNum(bInDesc) * sizeof(float));
        stat = cnnlTranspose_v2(context->cnnlHandle(), opDesc, bInDesc, bData,
                                bDesc, bDataOut, wsData, wsSize);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        // get outputs
        checkCnnlError(cnnlCreateTensorDescriptor(&cInDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            cInDesc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_FLOAT, 4, outputArray));

        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, output));

        cnnlConvolutionForwardAlgo_t algo;
        cnnlGetConvolutionForwardAlgorithm(context->cnnlHandle(), convDesc,
                                           aDesc, bDesc, cInDesc,
                                           CNNL_CONVOLUTION_FWD_FASTEST, &algo);

        cnnlGetConvolutionForwardWorkspaceSize(context->cnnlHandle(), aDesc,
                                               bDesc, cInDesc, NULL, convDesc,
                                               algo, &wsSize);
        wsData = context->getWorkspace(wsSize);
        BangPtr cDataIn = context->getWorkspace(
            cnnlGetTensorElementNum(cInDesc) * sizeof(float));

        stat = cnnlConvolutionForward(
            context->cnnlHandle(), convDesc, algo, NULL, aDesc, aData, bDesc,
            bData, NULL, NULL, wsData, wsSize, NULL, cInDesc, cDataIn);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        int cPermute[4] = {0, 3, 1, 2};
        cnnlTransposeDescriptor_t opOutDesc;
        checkCnnlError(cnnlCreateTransposeDescriptor(&opOutDesc));
        checkCnnlError(cnnlSetTransposeDescriptor(opOutDesc, 4, cPermute));

        cnnlGetTransposeWorkspaceSize(context->cnnlHandle(), cInDesc, opOutDesc,
                                      &wsSize);
        wsData = context->getWorkspace(wsSize);

        stat = cnnlTranspose_v2(context->cnnlHandle(), opOutDesc, cInDesc,
                                cDataIn, cDesc, cData, wsData, wsSize);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        // Destories in BANG does not require sync. But cnnl does not state
        // whether sync is required before destories.
        checkCnnlError(cnnlDestroyTensorDescriptor(aInDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(bInDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cInDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(bDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
        checkCnnlError(cnnlDestroyConvolutionDescriptor(convDesc));
        checkCnnlError(cnnlDestroyTransposeDescriptor(opDesc));
        checkCnnlError(cnnlDestroyTransposeDescriptor(opOutDesc));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::Conv, DataType::Float32, ConvCnnl,
                "Conv_cnnl_BANG_Float32");
}; // namespace infini

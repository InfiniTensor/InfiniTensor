#include "operators/conv.h"
#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"

namespace infini {
class ConvCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ConvBaseObj>(_op);
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

        int inputs0[4] = {n, c, h, w};
        int inputs0Array[4] = {n, h, w, c};
        int inputs1[4] = {f, c, r, s};
        int inputs1Array[4] = {f, r, s, c};
        int output[4] = {n, c, h, w};
        int outputArray[4] = {n, h, w, c};

        if (op->getOpType() == OpType::Conv) {
            cnnlTensorDescriptor_t aInDesc, aDesc, bInDesc, bDesc, cInDesc,
                cDesc;
            // get inputs
            checkCnnlError(cnnlCreateTensorDescriptor(&aInDesc));
            checkCnnlError(cnnlSetTensorDescriptor(
                aInDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_FLOAT, 4, inputs0));

            checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
            checkCnnlError(cnnlSetTensorDescriptor(
                aDesc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_FLOAT, 4, inputs0Array));

            checkCnnlError(cnnlCreateTensorDescriptor(&bInDesc));
            checkCnnlError(cnnlSetTensorDescriptor(
                bInDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_FLOAT, 4, inputs1));

            checkCnnlError(cnnlCreateTensorDescriptor(&bDesc));
            checkCnnlError(cnnlSetTensorDescriptor(
                bDesc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_FLOAT, 4, inputs1Array));

            int permute[4] = {0, 2, 3, 1};
            cnnlTransposeDescriptor_t opDesc;
            checkCnnlError(cnnlCreateTransposeDescriptor(&opDesc));
            checkCnnlError(cnnlSetTransposeDescriptor(opDesc, 4, permute));

            size_t wsSize;
            cnnlGetTransposeWorkspaceSize(context->cnnlHandle(), aInDesc,
                                          opDesc, &wsSize);
            BangPtr wsData = context->getWorkspace(wsSize);
            BangPtr aDataOut = context->getWorkspace(
                cnnlGetTensorElementNum(aInDesc) * sizeof(float));
            cnnlStatus_t stat =
                cnnlTranspose_v2(context->cnnlHandle(), opDesc, aInDesc, aData,
                                 aDesc, aDataOut, wsData, wsSize);
            if (stat != CNNL_STATUS_SUCCESS)
                return;

            cnnlGetTransposeWorkspaceSize(context->cnnlHandle(), bInDesc,
                                          opDesc, &wsSize);
            wsData = context->getWorkspace(wsSize);
            BangPtr bDataOut = context->getWorkspace(
                cnnlGetTensorElementNum(bInDesc) * sizeof(float));
            stat = cnnlTranspose_v2(context->cnnlHandle(), opDesc, bInDesc,
                                    bData, bDesc, bDataOut, wsData, wsSize);
            if (stat != CNNL_STATUS_SUCCESS)
                return;

            // get outputs
            checkCnnlError(cnnlCreateTensorDescriptor(&cInDesc));
            checkCnnlError(cnnlSetTensorDescriptor(
                cInDesc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_FLOAT, 4, outputArray));

            checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
            checkCnnlError(cnnlSetTensorDescriptor(
                cDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_FLOAT, 4, output));

            cnnlConvolutionForwardAlgo_t algo;
            cnnlGetConvolutionForwardAlgorithm(
                context->cnnlHandle(), convDesc, aDesc, bDesc, cInDesc,
                CNNL_CONVOLUTION_FWD_FASTEST, &algo);

            cnnlGetConvolutionForwardWorkspaceSize(context->cnnlHandle(), aDesc,
                                                   bDesc, cInDesc, NULL,
                                                   convDesc, algo, &wsSize);
            wsData = context->getWorkspace(wsSize);
            BangPtr cDataIn = context->getWorkspace(
                cnnlGetTensorElementNum(cInDesc) * sizeof(float));

            stat = cnnlConvolutionForward(context->cnnlHandle(), convDesc, algo,
                                          NULL, aDesc, aDataOut, bDesc,
                                          bDataOut, NULL, NULL, wsData, wsSize,
                                          NULL, cInDesc, cDataIn);
            if (stat != CNNL_STATUS_SUCCESS)
                return;

            int cPermute[4] = {0, 3, 1, 2};
            cnnlTransposeDescriptor_t opOutDesc;
            checkCnnlError(cnnlCreateTransposeDescriptor(&opOutDesc));
            checkCnnlError(cnnlSetTransposeDescriptor(opOutDesc, 4, cPermute));

            cnnlGetTransposeWorkspaceSize(context->cnnlHandle(), cInDesc,
                                          opOutDesc, &wsSize);
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
        } else if (op->getOpType() == OpType::ConvNHWC) {
            cnnlTensorDescriptor_t aDesc, bDesc, cDesc;
            checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
            checkCnnlError(cnnlSetTensorDescriptor(
                aDesc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_FLOAT, 4, inputs0Array));

            checkCnnlError(cnnlCreateTensorDescriptor(&bDesc));
            checkCnnlError(cnnlSetTensorDescriptor(
                bDesc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_FLOAT, 4, inputs1Array));

            checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
            checkCnnlError(cnnlSetTensorDescriptor(
                cDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_FLOAT, 4, outputArray));

            cnnlConvolutionForwardAlgo_t algo;
            cnnlGetConvolutionForwardAlgorithm(
                context->cnnlHandle(), convDesc, aDesc, bDesc, cDesc,
                CNNL_CONVOLUTION_FWD_FASTEST, &algo);

            size_t wsSize;
            cnnlGetConvolutionForwardWorkspaceSize(context->cnnlHandle(), aDesc,
                                                   bDesc, cDesc, NULL, convDesc,
                                                   algo, &wsSize);
            BangPtr wsData = context->getWorkspace(wsSize);

            cnnlStatus_t stat = cnnlConvolutionForward(
                context->cnnlHandle(), convDesc, algo, NULL, aDesc, aData,
                bDesc, bData, NULL, NULL, wsData, wsSize, NULL, cDesc, cData);
            if (stat != CNNL_STATUS_SUCCESS)
                return;

            // Destories in BANG does not require sync. But cnnl does not state
            // whether sync is required before destories.
            checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
            checkCnnlError(cnnlDestroyTensorDescriptor(bDesc));
            checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
            checkCnnlError(cnnlDestroyConvolutionDescriptor(convDesc));
        }
    }
};

REGISTER_KERNEL(Device::BANG, OpType::Conv, DataType::Float32, ConvCnnl,
                "Conv_cnnl_BANG_Float32");
}; // namespace infini

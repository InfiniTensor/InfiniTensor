#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"
#include "operators/conv.h"

namespace infini {
class ConvBackwardFilterCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ConvBackwardFilterObj>(_op);
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

        cnnlTensorDescriptor_t aDesc, bDesc, cDesc, aDescTrans, bDescTrans,
            cDescTrans;
        auto dimInputs0 = op->getInputs(0)->getDims();
        auto dimInputs1 = op->getInputs(1)->getDims();
        auto dimOutput = op->getOutput()->getDims();

        if (dimInputs0.size() != 4)
            IT_TODO_HALT();
        if (dimInputs1.size() != 4)
            IT_TODO_HALT();
        if (dimOutput.size() != 4)
            IT_TODO_HALT();

        int inputs0Array[4] = {dimInputs0[0], dimInputs0[1], dimInputs0[2],
                               dimInputs0[3]};
        int inputs1Array[4] = {dimInputs1[0], dimInputs1[1], dimInputs1[2],
                               dimInputs1[3]};
        int outputArray[4] = {dimOutput[0], dimOutput[1], dimOutput[2],
                              dimOutput[3]};

        int inputs0ArrayTrans[4] = {dimInputs0[0], dimInputs0[2], dimInputs0[3],
                                    dimInputs0[1]};
        int inputs1ArrayTrans[4] = {dimInputs1[0], dimInputs1[2], dimInputs1[3],
                                    dimInputs1[1]};
        int outputArrayTrans[4] = {dimOutput[0], dimOutput[2], dimOutput[3],
                                   dimOutput[1]};

        int transMode[4] = {0, 2, 3, 1};
        cnnlTransposeDescriptor_t transDesc;
        checkCnnlError(cnnlCreateTransposeDescriptor(&transDesc));
        checkCnnlError(cnnlSetTransposeDescriptor(transDesc, 4, transMode));

        // get inputs
        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            aDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_FLOAT, 4, inputs0Array));

        checkCnnlError(cnnlCreateTensorDescriptor(&aDescTrans));
        checkCnnlError(cnnlSetTensorDescriptor(aDescTrans, CNNL_LAYOUT_NHWC,
                                               CNNL_DTYPE_FLOAT, 4,
                                               inputs0ArrayTrans));

        size_t wsTrans1Size = dimInputs0[0] * dimInputs0[1] * dimInputs0[2] *
                              dimInputs0[3] * sizeof(float);
        BangPtr wsTrans1Data = context->getWorkspace(wsTrans1Size);

        cnnlStatus_t stat =
            cnnlTranspose(context->cnnlHandle(), transDesc, aDesc, aData,
                          aDescTrans, wsTrans1Data);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        checkCnnlError(cnnlCreateTensorDescriptor(&bDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            bDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_FLOAT, 4, inputs1Array));

        checkCnnlError(cnnlCreateTensorDescriptor(&bDescTrans));
        checkCnnlError(cnnlSetTensorDescriptor(bDescTrans, CNNL_LAYOUT_NHWC,
                                               CNNL_DTYPE_FLOAT, 4,
                                               inputs1ArrayTrans));

        size_t wsTrans2Size = dimInputs1[0] * dimInputs1[1] * dimInputs1[2] *
                              dimInputs1[3] * sizeof(float);
        BangPtr wsTrans2Data = context->getWorkspace(wsTrans2Size);

        stat = cnnlTranspose(context->cnnlHandle(), transDesc, bDesc, bData,
                             bDescTrans, wsTrans2Data);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        // get outputs
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            cDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_FLOAT, 4, outputArray));

        checkCnnlError(cnnlCreateTensorDescriptor(&cDescTrans));
        checkCnnlError(cnnlSetTensorDescriptor(cDescTrans, CNNL_LAYOUT_NHWC,
                                               CNNL_DTYPE_FLOAT, 4,
                                               outputArrayTrans));

        size_t wsTrans3Size = dimOutput[0] * dimOutput[1] * dimOutput[2] *
                              dimOutput[3] * sizeof(float);
        BangPtr wsTrans3Data = context->getWorkspace(wsTrans3Size);

        cnnlConvolutionBwdFilterAlgo_t algo;
        cnnlGetConvolutionBackwardFilterAlgorithm(
            context->cnnlHandle(), convDesc, aDescTrans, bDescTrans, cDescTrans,
            CNNL_CONVOLUTION_BWD_FILTER_FASTEST, &algo);

        size_t wsSize;
        cnnlGetConvolutionBackwardFilterWorkspaceSize(
            context->cnnlHandle(), aDescTrans, bDescTrans, cDescTrans, convDesc,
            algo, &wsSize);
        BangPtr wsData = context->getWorkspace(wsSize);

        stat = cnnlConvolutionBackwardFilter(
            context->cnnlHandle(), NULL, aDescTrans, wsTrans1Data, bDescTrans,
            wsTrans2Data, convDesc, algo, wsData, wsSize, NULL, cDescTrans,
            wsTrans3Data);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        int transMode2[4] = {0, 3, 1, 2};
        cnnlTransposeDescriptor_t transOutputDesc;
        checkCnnlError(cnnlCreateTransposeDescriptor(&transOutputDesc));
        checkCnnlError(
            cnnlSetTransposeDescriptor(transOutputDesc, 4, transMode2));

        stat = cnnlTranspose(context->cnnlHandle(), transOutputDesc, cDescTrans,
                             wsTrans3Data, cDesc, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        // Destories in BANG does not require sync. But cnnl does not state
        // whether sync is required before destories.
        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(bDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(aDescTrans));
        checkCnnlError(cnnlDestroyTensorDescriptor(bDescTrans));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDescTrans));
        checkCnnlError(cnnlDestroyTransposeDescriptor(transDesc));
        checkCnnlError(cnnlDestroyTransposeDescriptor(transOutputDesc));
        checkCnnlError(cnnlDestroyConvolutionDescriptor(convDesc));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::ConvBackwardFilter, DataType::Float32,
                ConvBackwardFilterCnnl, "ConvBackwardFilter_cnnl_BANG_Float32");
}; // namespace infini

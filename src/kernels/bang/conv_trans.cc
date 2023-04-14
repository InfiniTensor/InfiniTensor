#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"
#include "operators/conv.h"

namespace infini {
class ConvTransCnnl : public BangKernelWithoutConfig {
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

        cnnlTensorDescriptor_t aDesc, bDesc, cDesc;
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
        int inputs1[4] = {dimInputs1[0], dimInputs1[1], dimInputs1[2],
                          dimInputs1[3]};
        int output[4] = {dimOutput[0], dimOutput[1], dimOutput[2],
                         dimOutput[3]};

        // get inputs
        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, inputs0));
        checkCnnlError(cnnlCreateTensorDescriptor(&bDesc));
        checkCnnlError(cnnlSetTensorDescriptor(bDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, inputs1));
        // get outputs
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, output));

        cnnlConvolutionBwdDataAlgo_t algo;
        cnnlGetConvolutionBackwardDataAlgorithm(
            context->cnnlHandle(), aDesc, bDesc, convDesc, cDesc,
            CNNL_CONVOLUTION_BWD_DATA_FASTEST, &algo);
        size_t wsSize;
        cnnlGetConvolutionBackwardDataWorkspaceSize(context->cnnlHandle(),
                                                    aDesc, bDesc, convDesc,
                                                    cDesc, algo, &wsSize);
        BangPtr wsData = context->getWorkspace(wsSize);

        cnnlStatus_t stat = cnnlConvolutionBackwardData(
            context->cnnlHandle(), NULL, aDesc, aData, bDesc, bData, convDesc,
            algo, wsData, wsSize, NULL, cDesc, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        // Destories in BANG does not require sync. But cnnl does not state
        // whether sync is required before destories.
        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(bDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
        checkCnnlError(cnnlDestroyConvolutionDescriptor(convDesc));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::ConvTrans, DataType::Float32,
                ConvTransCnnl, "ConvTrans_cnnl_BANG_Float32");
}; // namespace infini

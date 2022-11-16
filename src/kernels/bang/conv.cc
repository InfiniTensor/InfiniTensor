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

        int pad[4] = {ph,ph,pw,pw};
        int stride[2] = {sh,sw};
        int dilation[2] = {dh,dw};

        cnnlConvolutionDescriptor_t convDesc;
        checkCnnlError(cnnlCreateConvolutionDescriptor(&convDesc));
        checkCnnlError(cnnlSetConvolutionDescriptor(convDesc, 4, pad, stride, dilation, g, CNNL_DTYPE_FLOAT));

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const biasData = (op->getInputs(2)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        cnnlTensorDescriptor_t aDesc, bDesc, cDesc, biasDesc;
        auto dimInputs0 = op->getInputs(0)->getDims();
        auto dimInputs1 = op->getInputs(1)->getDims();
        auto dimInputs2 = op->getInputs(2)->getDims();
        auto dimOutput = op->getOutput()->getDims();
        if (dimInputs0.size() != 4)
            IT_TODO_HALT();
        if (dimInputs1.size() != 4)
            IT_TODO_HALT();
        if (dimInputs2.size() != 4)
            IT_TODO_HALT();
        if (dimOutput.size() != 4)
            IT_TODO_HALT();

        // 这里是无奈之举，由于 cnnl 中 conv 仅支持 NHWC layout
        // 而 InfiniTensor 这里的 conv 与 cudnn 强相关于 NCHW layout
        // 我这样做，只是为了让程序跑起来，方便掐 conv 性能
        // 使用该算子的用户应当知道，这样做之后，计算结果完全是错误的。
        // 如果想要正确的结果，应该在适当的地方插入 Transpose。
        int inputs0Array[4] = {dimInputs0[0], dimInputs0[2], dimInputs0[3], dimInputs0[1]};
        int inputs1Array[4] = {dimInputs1[0], dimInputs1[2], dimInputs1[3], dimInputs1[1]};
        int inputs2Array[4] = {dimInputs2[0], dimInputs2[2], dimInputs2[3], dimInputs2[1]};
        int outputArray[4] = {dimOutput[0], dimOutput[2], dimOutput[3], dimOutput[1]};

        // get inputs
        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_NHWC,
                                               CNNL_DTYPE_FLOAT, 4, inputs0Array));

        checkCnnlError(cnnlCreateTensorDescriptor(&bDesc));
        checkCnnlError(cnnlSetTensorDescriptor(bDesc, CNNL_LAYOUT_NHWC,
                                               CNNL_DTYPE_FLOAT, 4, inputs1Array));

        checkCnnlError(cnnlCreateTensorDescriptor(&biasDesc));
        checkCnnlError(cnnlSetTensorDescriptor(biasDesc, CNNL_LAYOUT_NHWC,
                                               CNNL_DTYPE_FLOAT, 4, inputs2Array));
        // get outputs
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_NHWC,
                                               CNNL_DTYPE_FLOAT, 4, outputArray));

        cnnlConvolutionForwardAlgo_t algo;
        cnnlGetConvolutionForwardAlgorithm(context->cnnlHandle(), convDesc, aDesc, bDesc, cDesc, CNNL_CONVOLUTION_FWD_FASTEST, &algo);

        size_t wsSize;
        cnnlGetConvolutionForwardWorkspaceSize(context->cnnlHandle(), aDesc, bDesc, cDesc, biasDesc, convDesc, algo, &wsSize);
        BangPtr wsData = context->getWorkspace(wsSize);

        cnnlStatus_t stat = cnnlConvolutionForward(context->cnnlHandle(), convDesc, algo, NULL,
                                                   aDesc, aData, bDesc, bData, biasDesc, biasData, wsData, wsSize, NULL, cDesc, cData);
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

REGISTER_KERNEL(Device::BANG, OpType::Conv, DataType::Float32, ConvCnnl,
                "Conv_cnnl_BANG_Float32");
}; // namespace infini

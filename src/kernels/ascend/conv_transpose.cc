#include "aclnnop/level2/aclnn_convolution.h"
#include "ascend/ascend_kernel_without_config.h"
#include "ascend/ascend_runtime.h"
#include "operators/conv.h"

namespace infini {

class ConvTransAclnn : public ASCENDKernelWithoutConfig {

    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ConvTransposed2dObj>(_op);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);

        const auto [ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
        const auto [oph, opw] = op->getOutputPadding();
        const auto [n, c, h, w, f, r, s] = op->getNCHWFRS();
        const int cpg = op->getChannelPerGroup();
        const int g = c / cpg;

        std::vector<int64_t> pads = {ph, pw};
        // std::vector<int64_t> ksize = {r, s};
        std::vector<int64_t> stride = {sh, sw};
        std::vector<int64_t> dilation = {dh, dw};
        std::vector<int64_t> outputPadding = {oph, opw};

        aclIntArray *convpads = aclCreateIntArray(pads.data(), pads.size());
        aclIntArray *convstride =
            aclCreateIntArray(stride.data(), stride.size());
        aclIntArray *convdilation =
            aclCreateIntArray(dilation.data(), dilation.size());
        aclIntArray *convOutputpadding =
            aclCreateIntArray(outputPadding.data(), outputPadding.size());

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        // void *const biasData = (op->getBias()->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto inputD = op->getInputs(0)->getDims();
        auto inputS = op->getInputs(0)->getStride();
        auto weightD = op->getInputs(1)->getDims();
        auto weightS = op->getInputs(1)->getStride();
        // auto biasD = op->getBias()->getDims();
        // auto biasS = op->getBias()->getStride();
        auto outD = op->getOutput()->getDims();
        auto outS = op->getOutput()->getStride();

        std::vector<int64_t> inputDim = castTo64(inputD);
        std::vector<int64_t> inputStride = castTo64(inputS);
        std::vector<int64_t> weightDim = castTo64(weightD);
        std::vector<int64_t> weightStride = castTo64(weightS);
        std::vector<int64_t> outputDim = castTo64(outD);
        std::vector<int64_t> outputStride = castTo64(outS);

        auto inputTensor =
            aclCreateTensor(inputDim.data(), inputDim.size(), ACL_FLOAT,
                            inputStride.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                            inputDim.data(), inputDim.size(), aData);
        auto weightTensor =
            aclCreateTensor(weightDim.data(), weightDim.size(), ACL_FLOAT,
                            weightStride.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                            weightDim.data(), weightDim.size(), bData);
        auto outputTensor =
            aclCreateTensor(outputDim.data(), outputDim.size(), ACL_FLOAT,
                            outputStride.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                            outputDim.data(), outputDim.size(), cData);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;

        auto ret = aclnnConvolutionGetWorkspaceSize(
            inputTensor, weightTensor, nullptr, convstride, convpads,
            convdilation, true, convOutputpadding, int64_t(g), outputTensor,
            int8_t(1), &workspaceSize, &executor);
        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = context->getWorkspace(workspaceSize);
        }
        assert(ret == ACL_SUCCESS);
        ret = aclnnConvolution(workspaceAddr, workspaceSize, executor,
                               context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);

        ret = aclrtSynchronizeStream(context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);

        // aclDestroyTensor(inputTensor);
        // aclDestroyTensor(weightTensor);
        // aclDestroyTensor(outputTensor);

        return;
    }
};

REGISTER_KERNEL(Device::ASCEND, OpType::ConvTranspose, ConvTransAclnn,
                "ConvTrans_ASCEND_float");
}; // namespace infini

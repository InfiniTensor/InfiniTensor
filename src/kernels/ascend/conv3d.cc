#include "aclnnop/level2/aclnn_convolution.h"
#include "ascend/ascend_kernel_without_config.h"
#include "ascend/ascend_runtime.h"
#include "operators/conv.h"

namespace infini {

class Conv3dAclnn : public ASCENDKernelWithoutConfig {

    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<Conv3dObj>(_op);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);

        const auto [pd, ph, pw, sd, sh, sw, dd, dh, dw] =
            op->getPadStrideDilation();
        const auto [n, c, d, h, w, f, t, r, s] = op->getNCDHWFTS();
        const int cpg = op->getChannelPerGroup();
        const int g = c / cpg;

        std::vector<int64_t> pads = {pd, ph, pw};
        std::vector<int64_t> stride = {sd, sh, sw};
        std::vector<int64_t> dilation = {dd, dh, dw};
        std::vector<int64_t> outputPadding = {sd - 1, sh - 1, sw - 1};

        aclIntArray *convpads = aclCreateIntArray(pads.data(), pads.size());
        aclIntArray *convstride =
            aclCreateIntArray(stride.data(), stride.size());
        aclIntArray *convdilation =
            aclCreateIntArray(dilation.data(), dilation.size());
        aclIntArray *convOutputpadding =
            aclCreateIntArray(outputPadding.data(), outputPadding.size());

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto inputD = op->getInputs(0)->getDims();
        auto inputS = op->getInputs(0)->getStride();
        auto weightD = op->getInputs(1)->getDims();
        auto weightS = op->getInputs(1)->getStride();
        auto outD = op->getOutput()->getDims();
        auto outS = op->getOutput()->getStride();

        std::vector<int64_t> inputDim = castTo64(inputD);
        std::vector<int64_t> inputStride = castTo64(inputS);
        std::vector<int64_t> weightDim = castTo64(weightD);
        std::vector<int64_t> weightStride = castTo64(weightS);
        std::vector<int64_t> outputDim = castTo64(outD);
        std::vector<int64_t> outputStride = castTo64(outS);

        auto aclDataType = aclnnDataTypeConvert(op->getDType());

        auto inputTensor =
            aclCreateTensor(inputDim.data(), inputDim.size(), aclDataType,
                            inputStride.data(), 0, aclFormat::ACL_FORMAT_NCDHW,
                            inputDim.data(), inputDim.size(), aData);
        auto weightTensor =
            aclCreateTensor(weightDim.data(), weightDim.size(), aclDataType,
                            weightStride.data(), 0, aclFormat::ACL_FORMAT_NCDHW,
                            weightDim.data(), weightDim.size(), bData);
        auto outputTensor =
            aclCreateTensor(outputDim.data(), outputDim.size(), aclDataType,
                            outputStride.data(), 0, aclFormat::ACL_FORMAT_NCDHW,
                            outputDim.data(), outputDim.size(), cData);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;

        auto ret = aclnnConvolutionGetWorkspaceSize(
            inputTensor, weightTensor, nullptr, convstride, convpads,
            convdilation, false, convOutputpadding, int64_t(g), outputTensor,
            int8_t(1), &workspaceSize, &executor);
        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = context->getWorkspace(workspaceSize);
        }
        auto tmp_err_msg = aclGetRecentErrMsg();
        if (tmp_err_msg != NULL) {
            printf(" ERROR Message : %s \n ", tmp_err_msg);
        }
        // printf("ret is %d\n", ret);
        assert(ret == ACL_SUCCESS);
        ret = aclnnConvolution(workspaceAddr, workspaceSize, executor,
                               context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);

        // aclDestroyTensor(inputTensor);
        // aclDestroyTensor(weightTensor);
        // aclDestroyTensor(outputTensor);

        return;
    }
};

REGISTER_KERNEL(Device::ASCEND, OpType::Conv3d, Conv3dAclnn,
                "conv3d_ASCEND_float");
}; // namespace infini

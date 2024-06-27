#include "operators/resize.h"
#include "aclnnop/level2/aclnn_resize.h"
#include "ascend/ascend_kernel_without_config.h"
#include "ascend/ascend_runtime.h"

namespace infini {
class ResizeAclnn : public ASCENDKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ResizeObj>(_op);
        // IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);

        int nDims = op->getInputs(0)->getRank();
        if (nDims > 4)
            IT_TODO_HALT();

        vector<float> scalesData = op->getScales();

        const char *mode;
        switch (op->getMode()) {
        case ResizeObj::ECoeffMode::nearest:
            mode = "nearest";
            break;
        case ResizeObj::ECoeffMode::linear:
            mode = "bilinear";
            break;
        default:
            IT_TODO_HALT();
        }

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto a = op->getInputs(0)->getDims();
        auto aS = op->getInputs(0)->getStride();
        auto c = op->getOutput()->getDims();
        auto cS = op->getOutput()->getStride();

        std::vector<int64_t> aDim = castTo64(a);
        std::vector<int64_t> aStride = castTo64(aS);
        std::vector<int64_t> cDim = castTo64(c);
        std::vector<int64_t> cStride = castTo64(cS);

        auto aclDataType = aclnnDataTypeConvert(op->getDType());

        auto self = aclCreateTensor(
            aDim.data(), aDim.size(), aclDataType, aStride.data(), 0,
            aclFormat::ACL_FORMAT_NCHW, aDim.data(), aDim.size(), aData);

        auto output = aclCreateTensor(
            cDim.data(), cDim.size(), aclDataType, cStride.data(), 0,
            aclFormat::ACL_FORMAT_NCHW, cDim.data(), cDim.size(), cData);

        aclFloatArray *scales = nullptr;
        scales = aclCreateFloatArray(scalesData.data(), scalesData.size());
        CHECK_RET(scales != nullptr,
                  LOG_PRINT("aclCreateFloatArray failed.\n"));

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;

        auto ret = aclnnResizeGetWorkspaceSize(self, scales, mode, output,
                                               &workspaceSize, &executor);
        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = context->getWorkspace(workspaceSize);
        }
        CHECK_RET(
            ret == ACL_SUCCESS,
            LOG_PRINT("aclnnResizeGetWorkspaceSize failed. ERROR: %d\n", ret));

        ret = aclnnResize(workspaceAddr, workspaceSize, executor,
                          context->ASCENDHandle());
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("aclnnResize failed. ERROR: %d\n", ret));
        // auto tmp_err_msg = aclGetRecentErrMsg();
        // if (tmp_err_msg != NULL) {
        //     printf(" ERROR Message : %s \n ", tmp_err_msg);
        // }

        return;
    }
};

REGISTER_KERNEL(Device::ASCEND, OpType::Resize, ResizeAclnn, "Resize_ASCEND");

} // namespace infini

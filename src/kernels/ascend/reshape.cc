#include "operators/reshape.h"
#include "aclnnop/level2/aclnn_copy.h"
#include "ascend/ascend_kernel_without_config.h"
#include "ascend/ascend_runtime.h"

namespace infini {
class CopyAclnn : public ASCENDKernelWithoutConfig {
    void compute(const Operator &op,
                 const RuntimeObj *_context) const override {
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto aD = op->getInputs(0)->getDims();
        auto aS = op->getInputs(0)->getStride();

        std::vector<int64_t> aDim = castTo64(aD);
        std::vector<int64_t> aStride = castTo64(aS);

        auto aclDataType = aclnnDataTypeConvert(op->getDType());

        auto srcTensor = aclCreateTensor(
            aDim.data(), aDim.size(), aclDataType, aStride.data(), 0,
            aclFormat::ACL_FORMAT_ND, aDim.data(), aDim.size(), aData);
        auto outputTensor = aclCreateTensor(
            aDim.data(), aDim.size(), aclDataType, aStride.data(), 0,
            aclFormat::ACL_FORMAT_ND, aDim.data(), aDim.size(), cData);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;

        auto ret = aclnnInplaceCopyGetWorkspaceSize(outputTensor, srcTensor,
                                                    &workspaceSize, &executor);
        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = context->getWorkspace(workspaceSize);
        }
        assert(ret == ACL_SUCCESS);
        ret = aclnnInplaceCopy(workspaceAddr, workspaceSize, executor,
                               context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);

        return;
    }
};

REGISTER_KERNEL(Device::ASCEND, OpType::Reshape, CopyAclnn,
                "reshape_ASCEND_float");
REGISTER_KERNEL(Device::ASCEND, OpType::Unsqueeze, CopyAclnn,
                "unsqueeze_ASCEND_float");
REGISTER_KERNEL(Device::ASCEND, OpType::Squeeze, CopyAclnn,
                "squeeze_ASCEND_float");
REGISTER_KERNEL(Device::ASCEND, OpType::Flatten, CopyAclnn,
                "Flatten_ASCEND_float");
REGISTER_KERNEL(Device::ASCEND, OpType::Identity, CopyAclnn,
                "Identity_ASCEND_float");
}; // namespace infini

#include "operators/gather.h"
#include "aclnnop/level2/aclnn_gather_v2.h"
#include "ascend/ascend_kernel_without_config.h"
#include "ascend/ascend_runtime.h"

namespace infini {

class GatherAclnn : public ASCENDKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<GatherObj>(_op);
        IT_ASSERT(op->getInputs(1)->getDType() == DataType::Int32 ||
                  op->getInputs(1)->getDType() == DataType::Int64);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        int64_t axis = int64_t(op->getAxis());

        auto a = op->getInputs(0)->getDims();
        auto aS = op->getInputs(0)->getStride();
        auto b = op->getInputs(1)->getDims();
        auto bS = op->getInputs(1)->getStride();
        auto c = op->getOutput()->getDims();
        auto cS = op->getOutput()->getStride();

        std::vector<int64_t> aDim = castTo64(a);
        std::vector<int64_t> aStride = castTo64(aS);
        std::vector<int64_t> bDim = castTo64(b);
        std::vector<int64_t> bStride = castTo64(bS);
        std::vector<int64_t> cDim = castTo64(c);
        std::vector<int64_t> cStride = castTo64(cS);

        auto inputA = aclCreateTensor(
            aDim.data(), aDim.size(), ACL_FLOAT, aStride.data(), 0,
            aclFormat::ACL_FORMAT_ND, aDim.data(), aDim.size(), aData);

        auto inputB = aclCreateTensor(
            bDim.data(), bDim.size(),
            op->getInputs(1)->getDType() == DataType::Int32 ? ACL_INT32
                                                            : ACL_INT64,
            bStride.data(), 0, aclFormat::ACL_FORMAT_ND, bDim.data(),
            bDim.size(), bData);

        auto output = aclCreateTensor(
            cDim.data(), cDim.size(), ACL_FLOAT, cStride.data(), 0,
            aclFormat::ACL_FORMAT_ND, cDim.data(), cDim.size(), cData);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;

        auto ret = aclnnGatherV2GetWorkspaceSize(inputA, axis, inputB, output,
                                                 &workspaceSize, &executor);
        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = context->getWorkspace(workspaceSize);
        }
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("aclnnGatherV2GetWorkspaceSize failed. ERROR: %d\n",
                            ret));

        ret = aclnnGatherV2(workspaceAddr, workspaceSize, executor,
                            context->ASCENDHandle());
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("aclnnGatherV2 failed. ERROR: %d\n", ret));
        auto tmp_err_msg = aclGetRecentErrMsg();
        if (tmp_err_msg != NULL) {
            printf(" ERROR Message : %s \n ", tmp_err_msg);
        }

        ret = aclrtSynchronizeStream(context->ASCENDHandle());
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret));

        return;
    }
};

REGISTER_KERNEL(Device::ASCEND, OpType::Gather, GatherAclnn,
                "gather_ASCEND_float");
}; // namespace infini

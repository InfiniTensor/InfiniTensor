#include "operators/transpose.h"
#include "aclnnop/level2/aclnn_permute.h"
#include "ascend/ascend_kernel_without_config.h"
#include "ascend/ascend_runtime.h"

namespace infini {

class PermuteAclnn : public ASCENDKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<TransposeObj>(_op);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);

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

        auto _permute = op->getPermute();
        std::vector<int64_t> permute = castTo64(_permute);

        auto inputA = aclCreateTensor(
            aDim.data(), aDim.size(), ACL_FLOAT, aStride.data(), 0,
            aclFormat::ACL_FORMAT_ND, aDim.data(), aDim.size(), aData);
        aclIntArray *dims = aclCreateIntArray(permute.data(), permute.size());
        auto output = aclCreateTensor(
            cDim.data(), cDim.size(), ACL_FLOAT, cStride.data(), 0,
            aclFormat::ACL_FORMAT_ND, cDim.data(), cDim.size(), cData);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;

        auto ret = aclnnPermuteGetWorkspaceSize(inputA, dims, output,
                                                &workspaceSize, &executor);
        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = context->getWorkspace(workspaceSize);
        }
        assert(ret == ACL_SUCCESS);
        ret = aclnnPermute(workspaceAddr, workspaceSize, executor,
                           context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);

        ret = aclrtSynchronizeStream(context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);

        return;
    }
};

class DepthToSpaceAclnn : public ASCENDKernelWithoutConfig {

    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<DepthToSpaceObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto reshapeDim = op->getReshapeDim();
        auto reshapeStride = getStride(reshapeDim);
        auto transposeDim = op->getTransposeDim();
        auto transposeStride = getStride(transposeDim);

        std::vector<int64_t> aDim = castTo64(reshapeDim);
        std::vector<int64_t> aStride = castTo64(reshapeStride);
        std::vector<int64_t> cDim = castTo64(transposeDim);
        std::vector<int64_t> cStride = castTo64(transposeStride);

        auto mode = op->getMode();

        std::vector<int64_t> permute;
        if (mode == 0) {
            permute = {0, 3, 4, 1, 5, 2};
        } else {
            permute = {0, 1, 4, 2, 5, 3};
        }

        auto inputA = aclCreateTensor(
            aDim.data(), aDim.size(), ACL_FLOAT, aStride.data(), 0,
            aclFormat::ACL_FORMAT_ND, aDim.data(), aDim.size(), aData);
        aclIntArray *dims = aclCreateIntArray(permute.data(), permute.size());
        auto output = aclCreateTensor(
            cDim.data(), cDim.size(), ACL_FLOAT, cStride.data(), 0,
            aclFormat::ACL_FORMAT_ND, cDim.data(), cDim.size(), cData);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;

        auto ret = aclnnPermuteGetWorkspaceSize(inputA, dims, output,
                                                &workspaceSize, &executor);
        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = context->getWorkspace(workspaceSize);
        }
        assert(ret == ACL_SUCCESS);
        ret = aclnnPermute(workspaceAddr, workspaceSize, executor,
                           context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);

        ret = aclrtSynchronizeStream(context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);

        return;
    }
};

REGISTER_KERNEL(Device::ASCEND, OpType::Transpose, PermuteAclnn,
                "transpose_ASCEND_float");
REGISTER_KERNEL(Device::ASCEND, OpType::DepthToSpace, DepthToSpaceAclnn,
                "DepthToSpace_ASCEND_float");
}; // namespace infini

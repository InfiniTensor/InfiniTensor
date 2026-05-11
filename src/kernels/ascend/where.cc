#include "operators/where.h"
#include "aclnnop/level2/aclnn_s_where.h"
#include "ascend/ascend_kernel_without_config.h"
#include "ascend/ascend_runtime.h"

namespace infini {
class WhereAclnn : public ASCENDKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<WhereObj>(_op);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getInputs(2)->getRawDataPtr<void *>());
        void *const dData = (op->getOutput()->getRawDataPtr<void *>());

        auto aD = op->getInputs(0)->getDims();
        auto aS = op->getInputs(0)->getStride();
        auto bD = op->getInputs(1)->getDims();
        auto bS = op->getInputs(1)->getStride();
        auto cD = op->getInputs(2)->getDims();
        auto cS = op->getInputs(2)->getStride();
        auto dD = op->getOutput()->getDims();
        auto dS = op->getOutput()->getStride();

        if (aD.size() == 0) {
            aD.push_back(1);
        }
        if (bD.size() == 0) {
            bD.push_back(1);
        }
        if (cD.size() == 0) {
            cD.push_back(1);
        }
        if (dD.size() == 0) {
            dD.push_back(1);
        }

        std::vector<int64_t> aDim = castTo64(aD);
        std::vector<int64_t> aStride = castTo64(aS);
        std::vector<int64_t> bDim = castTo64(bD);
        std::vector<int64_t> bStride = castTo64(bS);
        std::vector<int64_t> cDim = castTo64(cD);
        std::vector<int64_t> cStride = castTo64(cS);
        std::vector<int64_t> dDim = castTo64(dD);
        std::vector<int64_t> dStride = castTo64(dS);

        auto aclDataType = aclnnDataTypeConvert(op->getDType());

        auto self = aclCreateTensor(aDim.data(), aDim.size(), aclDataType,
                                    aStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                                    aDim.data(), aDim.size(), aData);
        auto other = aclCreateTensor(
            bDim.data(), bDim.size(), aclDataType, bStride.data(), 0,
            aclFormat::ACL_FORMAT_ND, bDim.data(), bDim.size(), bData);
        auto condition = aclCreateTensor(
            cDim.data(), cDim.size(), aclDataType, cStride.data(), 0,
            aclFormat::ACL_FORMAT_ND, cDim.data(), cDim.size(), cData);
        auto out = aclCreateTensor(dDim.data(), dDim.size(), aclDataType,
                                   dStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                                   dDim.data(), dDim.size(), dData);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;

        auto ret = aclnnSWhereGetWorkspaceSize(condition, self, other, out,
                                               &workspaceSize, &executor);
        checkASCENDError(ret);

        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = context->getWorkspace(workspaceSize);
        }

        ret = aclnnSWhere(workspaceAddr, workspaceSize, executor,
                          context->ASCENDHandle());
        checkASCENDError(ret);

        aclDestroyTensor(self);
        aclDestroyTensor(other);
        aclDestroyTensor(condition);
        aclDestroyTensor(out);
    }
};

REGISTER_KERNEL(Device::ASCEND, OpType::Where, WhereAclnn, "Where_ASCEND");

}; // namespace infini

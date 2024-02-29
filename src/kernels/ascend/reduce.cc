#include "operators/reduce.h"
#include "aclnnop/aclnn_mean.h"
#include "aclnnop/aclnn_reduce_sum.h"
#include "ascend/ascend_kernel_without_config.h"
#include "ascend/ascend_runtime.h"

namespace infini {

class MeanAclnn : public ASCENDKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ReduceBaseObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto axes_set = op->getAxes();
        std::vector<int> axes;
        axes.assign(axes_set.begin(), axes_set.end());

        bool KeepDim = op->getKeepDims();

        auto a = op->getInputs(0)->getDims();
        auto aS = op->getInputs(0)->getStride();
        auto c = op->getOutput()->getDims();
        auto cS = op->getOutput()->getStride();

        std::vector<int64_t> aDim = castTo64(a);
        std::vector<int64_t> aStride = castTo64(aS);
        std::vector<int64_t> cDim = castTo64(c);
        std::vector<int64_t> cStride = castTo64(cS);
        std::vector<int64_t> axes_64 = castTo64(axes);

        auto inputA = aclCreateTensor(
            aDim.data(), aDim.size(), ACL_FLOAT, aStride.data(), 0,
            aclFormat::ACL_FORMAT_ND, aDim.data(), aDim.size(), aData);
        auto output = aclCreateTensor(
            cDim.data(), cDim.size(), ACL_FLOAT, cStride.data(), 0,
            aclFormat::ACL_FORMAT_ND, cDim.data(), cDim.size(), cData);
        aclIntArray *dim = aclCreateIntArray(axes_64.data(), axes_64.size());

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;

        auto ret = aclnnMeanV2GetWorkspaceSize(
            inputA, dim, KeepDim, true, output, &workspaceSize, &executor);
        assert(ret == ACL_SUCCESS);
        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = context->getWorkspace(workspaceSize);
        }
        assert(ret == ACL_SUCCESS);
        ret = aclnnMeanV2(workspaceAddr, workspaceSize, executor,
                          context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);

        ret = aclrtSynchronizeStream(context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);

        return;
    }
};

class ReduceSumAclnn : public ASCENDKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ReduceBaseObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto axes_set = op->getAxes();
        std::vector<int> axes;
        axes.assign(axes_set.begin(), axes_set.end());

        bool KeepDim = op->getKeepDims();

        auto a = op->getInputs(0)->getDims();
        auto aS = op->getInputs(0)->getStride();
        auto c = op->getOutput()->getDims();
        auto cS = op->getOutput()->getStride();

        std::vector<int64_t> aDim = castTo64(a);
        std::vector<int64_t> aStride = castTo64(aS);
        std::vector<int64_t> cDim = castTo64(c);
        std::vector<int64_t> cStride = castTo64(cS);
        std::vector<int64_t> axes_64 = castTo64(axes);

        auto inputA = aclCreateTensor(
            aDim.data(), aDim.size(), ACL_FLOAT, aStride.data(), 0,
            aclFormat::ACL_FORMAT_ND, aDim.data(), aDim.size(), aData);
        auto output = aclCreateTensor(
            cDim.data(), cDim.size(), ACL_FLOAT, cStride.data(), 0,
            aclFormat::ACL_FORMAT_ND, cDim.data(), cDim.size(), cData);
        aclIntArray *dim = aclCreateIntArray(axes_64.data(), axes_64.size());

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;

        auto ret = aclnnReduceSumGetWorkspaceSize(
            inputA, dim, KeepDim, ACL_FLOAT, output, &workspaceSize, &executor);
        assert(ret == ACL_SUCCESS);
        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = context->getWorkspace(workspaceSize);
        }
        assert(ret == ACL_SUCCESS);
        ret = aclnnReduceSum(workspaceAddr, workspaceSize, executor,
                             context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);

        ret = aclrtSynchronizeStream(context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);

        return;
    }
};

REGISTER_KERNEL(Device::ASCEND, OpType::ReduceMean, MeanAclnn,
                "reduceMean_ASCEND_float");
REGISTER_KERNEL(Device::ASCEND, OpType::ReduceSum, ReduceSumAclnn,
                "reduceSum_ASCEND_float");
}; // namespace infini


#include "operators/softmax.h"
#include "aclnnop/level2/aclnn_softmax.h"
#include "ascend/ascend_kernel_without_config.h"
#include "ascend/ascend_runtime.h"

namespace infini {
class SoftmaxAclnn : public ASCENDKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<SoftmaxObj>(_op);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        int64_t axis = int64_t(op->getAxis());

        auto a = op->getInputs(0)->getDims();
        auto aS = op->getInputs(0)->getStride();
        auto c = op->getInputs(0)->getDims();
        auto cS = op->getInputs(0)->getStride();

        std::vector<int64_t> aDim = MycastTo64(a);
        std::vector<int64_t> aStride = MycastTo64(aS);
        std::vector<int64_t> cDim = MycastTo64(c);
        std::vector<int64_t> cStride = MycastTo64(cS);

        auto input = aclCreateTensor(
            aDim.data(), aDim.size(), ACL_FLOAT, aStride.data(), 0,
            aclFormat::ACL_FORMAT_ND, aDim.data(), aDim.size(), aData);
        auto output = aclCreateTensor(
            cDim.data(), cDim.size(), ACL_FLOAT, cStride.data(), 0,
            aclFormat::ACL_FORMAT_ND, cDim.data(), cDim.size(), cData);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;

        auto ret = aclnnSoftmaxGetWorkspaceSize(input, axis, output,
                                                &workspaceSize, &executor);
        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = context->getWorkspace(workspaceSize);
        }
        assert(ret == ACL_SUCCESS);
        ret = aclnnSoftmax(workspaceAddr, workspaceSize, executor,
                           context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);

        ret = aclrtSynchronizeStream(context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);

        // aclDestroyTensor(input);
        // aclDestroyTensor(output);
        return;
    }
};

REGISTER_KERNEL(Device::ASCEND, OpType::Softmax, SoftmaxAclnn,
                "softmax_ASCEND_float");

}; // namespace infini

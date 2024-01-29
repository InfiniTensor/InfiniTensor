#include "operators/reshape.h"
#include "aclnnop/level2/aclnn_copy.h"
#include "ascend/ascend_kernel_without_config.h"
#include "ascend/ascend_runtime.h"

namespace infini {

class CopyAclnn : public ASCENDKernelWithoutConfig {

    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<MatmulObj>(_op);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto selfD = op->getInputs(0)->getDims();
        auto selfS = op->getInputs(0)->getStride();
        auto matD = op->getInputs(1)->getDims();
        auto matS = op->getInputs(1)->getStride();
        auto outD = op->getOutput()->getDims();
        auto outS = op->getOutput()->getStride();

        std::vector<int64_t> selfDim = MycastTo64(selfD);
        std::vector<int64_t> selfStride = MycastTo64(selfS);
        std::vector<int64_t> matDim = MycastTo64(matD);
        std::vector<int64_t> matStride = MycastTo64(matS);
        std::vector<int64_t> outputDim = MycastTo64(outD);
        std::vector<int64_t> outputStride = MycastTo64(outS);

        auto selfTensor = aclCreateTensor(
            selfDim.data(), selfDim.size(), ACL_FLOAT, selfStride.data(), 0,
            aclFormat::ACL_FORMAT_ND, selfDim.data(), selfDim.size(), aData);
        auto matTensor = aclCreateTensor(
            matDim.data(), matDim.size(), ACL_FLOAT, matStride.data(), 0,
            aclFormat::ACL_FORMAT_ND, matDim.data(), matDim.size(), bData);
        auto outputTensor =
            aclCreateTensor(outputDim.data(), outputDim.size(), ACL_FLOAT,
                            outputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                            outputDim.data(), outputDim.size(), cData);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;

        auto ret = aclnnMatmulGetWorkspaceSize(
            selfTensor, matTensor, outputTensor, 1, &workspaceSize, &executor);
        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = context->getWorkspace(workspaceSize);
        }
        assert(ret == ACL_SUCCESS);
        ret = aclnnMatmul(workspaceAddr, workspaceSize, executor,
                          context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);

        ret = aclrtSynchronizeStream(context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);

        // aclDestroyTensor(selfTensor);
        // aclDestroyTensor(matTensor);
        // aclDestroyTensor(outputTensor);

        return;
    }
};

REGISTER_KERNEL(Device::ASCEND, OpType::MatMul, MatmulAclnn,
                "matmul_ASCEND_float");
}; // namespace infini

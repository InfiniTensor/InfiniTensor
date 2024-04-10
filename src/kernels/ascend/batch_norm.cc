#include "operators/batch_norm.h"
#include "aclnnop/level2/aclnn_batch_norm.h"
#include "ascend/ascend_kernel_without_config.h"
#include "ascend/ascend_runtime.h"

namespace infini {

class BatchNormAclnn : public ASCENDKernelWithoutConfig {

    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<BatchNormObj>(_op);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);

        void *const inData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const outData = (op->getOutput()->getRawDataPtr<void *>());
        void *const meanData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const varData = (op->getInputs(2)->getRawDataPtr<void *>());
        void *const scaleData = (op->getInputs(3)->getRawDataPtr<void *>());
        void *const biasData = (op->getInputs(4)->getRawDataPtr<void *>());

        auto inD = op->getInputs(0)->getDims();
        auto inS = op->getInputs(0)->getStride();
        auto paraD = op->getInputs(1)->getDims();
        auto paraS = op->getInputs(1)->getStride();
        auto outD = op->getOutput()->getDims();
        auto outS = op->getOutput()->getStride();

        std::vector<int64_t> inputDim = castTo64(inD);
        std::vector<int64_t> inputStride = castTo64(inS);
        std::vector<int64_t> paraDim = castTo64(paraD);
        std::vector<int64_t> paraStride = castTo64(paraS);
        std::vector<int64_t> outputDim = castTo64(outD);
        std::vector<int64_t> outputStride = castTo64(outS);

        auto inputTensor =
            aclCreateTensor(inputDim.data(), inputDim.size(), ACL_FLOAT,
                            inputStride.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                            inputDim.data(), inputDim.size(), inData);
        auto outputTensor =
            aclCreateTensor(outputDim.data(), outputDim.size(), ACL_FLOAT,
                            outputStride.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                            outputDim.data(), outputDim.size(), outData);
        auto meanTensor = aclCreateTensor(
            paraDim.data(), paraDim.size(), ACL_FLOAT, paraStride.data(), 0,
            aclFormat::ACL_FORMAT_ND, paraDim.data(), paraDim.size(), meanData);
        auto varTensor = aclCreateTensor(
            paraDim.data(), paraDim.size(), ACL_FLOAT, paraStride.data(), 0,
            aclFormat::ACL_FORMAT_ND, paraDim.data(), paraDim.size(), varData);
        auto scaleTensor =
            aclCreateTensor(paraDim.data(), paraDim.size(), ACL_FLOAT,
                            paraStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                            paraDim.data(), paraDim.size(), scaleData);
        auto biasTensor = aclCreateTensor(
            paraDim.data(), paraDim.size(), ACL_FLOAT, paraStride.data(), 0,
            aclFormat::ACL_FORMAT_ND, paraDim.data(), paraDim.size(), biasData);
        auto savemeanTensor =
            aclCreateTensor(paraDim.data(), paraDim.size(), ACL_FLOAT,
                            paraStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                            paraDim.data(), paraDim.size(), scaleData);
        auto saveinvstdTensor = aclCreateTensor(
            paraDim.data(), paraDim.size(), ACL_FLOAT, paraStride.data(), 0,
            aclFormat::ACL_FORMAT_ND, paraDim.data(), paraDim.size(), biasData);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;

        auto ret = aclnnBatchNormGetWorkspaceSize(
            inputTensor, scaleTensor, biasTensor, meanTensor, varTensor, false,
            op->getMomentum(), op->getEps(), outputTensor, savemeanTensor,
            saveinvstdTensor, &workspaceSize, &executor);
        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = context->getWorkspace(workspaceSize);
        }
        assert(ret == ACL_SUCCESS);
        ret = aclnnBatchNorm(workspaceAddr, workspaceSize, executor,
                             context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);

        ret = aclrtSynchronizeStream(context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);

        // aclDestroyTensor(inputTensor);
        // aclDestroyTensor(outputTensor);
        // aclDestroyTensor(meanTensor);
        // aclDestroyTensor(varTensor);
        // aclDestroyTensor(scaleTensor);
        // aclDestroyTensor(biasTensor);
        // aclDestroyTensor(savemeanTensor);
        // aclDestroyTensor(saveinvstdTensor);

        return;
    }
};

REGISTER_KERNEL(Device::ASCEND, OpType::BatchNormalization, BatchNormAclnn,
                "batchnorm_ASCEND_float");
}; // namespace infini

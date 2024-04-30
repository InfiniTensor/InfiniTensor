#include "operators/instance_norm.h"
#include "aclnnop/level2/aclnn_layer_norm.h"
#include "ascend/ascend_kernel_without_config.h"
#include "ascend/ascend_runtime.h"
#include "operators/gather.h"

namespace infini {

class InstanceNormAclnn : public ASCENDKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<InstanceNormObj>(_op);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);

        void *const inputData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const weightData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const outputData = (op->getOutput()->getRawDataPtr<void *>());

        auto inputD = op->getInputs(0)->getDims();
        auto inputS = op->getInputs(0)->getStride();
        auto weightD = op->getInputs(1)->getDims();
        auto weightS = op->getInputs(1)->getStride();
        auto outD = op->getOutput()->getDims();
        auto outS = op->getOutput()->getStride();

        double eps = static_cast<double>(op->getEps());

        std::vector<int64_t> inputDim = castTo64(inputD);
        std::vector<int64_t> inputStride = castTo64(inputS);
        std::vector<int64_t> weightDim = castTo64(weightD);
        std::vector<int64_t> weightStride = castTo64(weightS);
        std::vector<int64_t> outputDim = castTo64(outD);
        std::vector<int64_t> outputStride = castTo64(outS);

        auto axis = 3;

        auto rank = static_cast<int>(inputDim.size());
        std::vector<int64_t> normalizedShape(rank - axis, 0);
        for (auto i = rank; i > axis; --i) {
            normalizedShape[i - 1 - axis] = inputDim[i - 1];
        }

        auto inputTensor =
            aclCreateTensor(inputDim.data(), inputDim.size(), ACL_FLOAT,
                            inputStride.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                            inputDim.data(), inputDim.size(), inputData);
        auto weightTensor =
            aclCreateTensor(weightDim.data(), weightDim.size(), ACL_FLOAT,
                            weightStride.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                            weightDim.data(), weightDim.size(), weightData);
        auto outputTensor =
            aclCreateTensor(outputDim.data(), outputDim.size(), ACL_FLOAT,
                            outputStride.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                            outputDim.data(), outputDim.size(), outputData);

        auto *normArray =
            aclCreateIntArray(normalizedShape.data(), normalizedShape.size());

        aclTensor *biasTensor = NULL;
        if (op->numInputs() == 3) {
            void *const biasData = (op->getInputs(2)->getRawDataPtr<void *>());

            auto biasD = op->getInputs(2)->getDims();
            auto biasS = op->getInputs(2)->getStride();
            std::vector<int64_t> biasDim = castTo64(biasD);
            std::vector<int64_t> biasStride = castTo64(biasS);

            biasTensor = aclCreateTensor(
                biasDim.data(), biasDim.size(), ACL_FLOAT, biasStride.data(), 0,
                aclFormat::ACL_FORMAT_NCHW, biasDim.data(), biasDim.size(),
                biasData);
        }

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;

        auto ret = aclnnLayerNormGetWorkspaceSize(
            inputTensor, normArray, weightTensor, biasTensor, eps, outputTensor,
            NULL, NULL, &workspaceSize, &executor);

        CHECK_RET(
            ret == ACL_SUCCESS,
            LOG_PRINT("aclnnLayerNormGetWorkspaceSize failed. ERROR: %d\n",
                      ret));
        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = context->getWorkspace(workspaceSize);
        }
        auto tmp_err_msg = aclGetRecentErrMsg();
        if (tmp_err_msg != NULL) {
            printf(" ERROR Message : %s \n ", tmp_err_msg);
        }
        ret = aclnnLayerNorm(workspaceAddr, workspaceSize, executor,
                             context->ASCENDHandle());
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("aclnnLayerNorm failed. ERROR: %d\n", ret));

        ret = aclrtSynchronizeStream(context->ASCENDHandle());
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret));

        return;
    }
};

REGISTER_KERNEL(Device::ASCEND, OpType::InstanceNormalization,
                InstanceNormAclnn, "InstanceNorm_ASCEND");

}; // namespace infini

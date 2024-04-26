#include "operators/matmul.h"
#include "aclnnop/level2/aclnn_gemm.h"
#include "aclnnop/level2/aclnn_matmul.h"
#include "ascend/ascend_kernel_without_config.h"
#include "ascend/ascend_runtime.h"

namespace infini {

class MatmulAclnn : public ASCENDKernelWithoutConfig {
    // unsupport trans for "gemm" whithou biasInput
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<MatmulObj>(_op);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);

        auto input_num = op->numInputs();

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        void *biasData = NULL;
        if (input_num > 2) {
            biasData = (op->getInputs(2)->getRawDataPtr<void *>());
        }

        auto selfD = op->getInputs(0)->getDims();
        auto selfS = op->getInputs(0)->getStride();
        auto matD = op->getInputs(1)->getDims();
        auto matS = op->getInputs(1)->getStride();
        auto outD = op->getOutput()->getDims();
        auto outS = op->getOutput()->getStride();
        std::vector<int> biasD;
        std::vector<int> biasS;
        if (input_num > 2) {
            biasD = op->getInputs(2)->getDims();
            biasS = op->getInputs(2)->getStride();
        }

        std::vector<int64_t> selfDim = castTo64(selfD);
        std::vector<int64_t> selfStride = castTo64(selfS);
        std::vector<int64_t> matDim = castTo64(matD);
        std::vector<int64_t> matStride = castTo64(matS);
        std::vector<int64_t> outputDim = castTo64(outD);
        std::vector<int64_t> outputStride = castTo64(outS);
        std::vector<int64_t> biasDim;
        std::vector<int64_t> biasStride;
        if (input_num > 2) {
            biasDim = castTo64(biasD);
            biasStride = castTo64(biasS);
        }

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
        aclTensor *biasTensor = NULL;
        if (input_num > 2) {
            biasTensor =
                aclCreateTensor(biasDim.data(), biasDim.size(), ACL_FLOAT,
                                biasStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                                biasDim.data(), biasDim.size(), biasData);
        }

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;

        if (input_num > 2) {
            float alpha = 1.0;
            float beta = 1.0;
            int32_t transA = op->getTransA();
            int32_t transB = op->getTransB();

            auto ret = aclnnGemmGetWorkspaceSize(
                selfTensor, matTensor, biasTensor, alpha, beta, int64_t(transA),
                int64_t(transB), outputTensor, 0, &workspaceSize, &executor);
            void *workspaceAddr = nullptr;
            if (workspaceSize > 0) {
                workspaceAddr = context->getWorkspace(workspaceSize);
            }
            // auto tmp_err_msg = aclGetRecentErrMsg();
            // if (tmp_err_msg != NULL) {
            //     printf(" ERROR Message : %s \n ", tmp_err_msg);
            // }
            CHECK_RET(
                ret == ACL_SUCCESS,
                LOG_PRINT("aclnnMatmulGetWorkspaceSize failed. ERROR: %d\n",
                          ret));
            ret = aclnnGemm(workspaceAddr, workspaceSize, executor,
                            context->ASCENDHandle());
            CHECK_RET(ret == ACL_SUCCESS,
                      LOG_PRINT("aclnnGemm failed. ERROR: %d\n", ret));
        } else {
            auto ret =
                aclnnMatmulGetWorkspaceSize(selfTensor, matTensor, outputTensor,
                                            1, &workspaceSize, &executor);
            void *workspaceAddr = nullptr;
            if (workspaceSize > 0) {
                workspaceAddr = context->getWorkspace(workspaceSize);
            }
            CHECK_RET(
                ret == ACL_SUCCESS,
                LOG_PRINT("aclnnMatmulGetWorkspaceSize failed. ERROR: %d\n",
                          ret));

            ret = aclnnMatmul(workspaceAddr, workspaceSize, executor,
                              context->ASCENDHandle());
            CHECK_RET(ret == ACL_SUCCESS,
                      LOG_PRINT("aclnnMatmul failed. ERROR: %d\n", ret));
        }
        auto ret = aclrtSynchronizeStream(context->ASCENDHandle());
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret));

        // aclDestroyTensor(selfTensor);
        // aclDestroyTensor(matTensor);
        // aclDestroyTensor(outputTensor);

        return;
    }
};

REGISTER_KERNEL(Device::ASCEND, OpType::MatMul, MatmulAclnn,
                "matmul_ASCEND_float");
}; // namespace infini

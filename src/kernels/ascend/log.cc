#include "aclnnop/level2/aclnn_log.h"
#include "aclnnop/level2/aclnn_log10.h"
#include "aclnnop/level2/aclnn_log2.h"
#include "ascend/ascend_kernel_without_config.h"
#include "ascend/ascend_runtime.h"
#include "operators/unary.h"

namespace infini {
class LogAclnn : public ASCENDKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<LogObj>(_op);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto type = op->getType();

        auto inputD = op->getInputs(0)->getDims();
        auto inputS = op->getInputs(0)->getStride();
        auto outD = op->getOutput()->getDims();
        auto outS = op->getOutput()->getStride();

        std::vector<int64_t> inputDim = castTo64(inputD);
        std::vector<int64_t> inputStride = castTo64(inputS);
        std::vector<int64_t> outputDim = castTo64(outD);
        std::vector<int64_t> outputStride = castTo64(outS);

        auto aclDataType = aclnnDataTypeConvert(op->getDType());
        auto inputTensor =
            aclCreateTensor(inputDim.data(), inputDim.size(), aclDataType,
                            inputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                            inputDim.data(), inputDim.size(), aData);
        auto outputTensor =
            aclCreateTensor(outputDim.data(), outputDim.size(), aclDataType,
                            outputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                            outputDim.data(), outputDim.size(), cData);
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;
        void *workspaceAddr = nullptr;
        aclnnStatus ret;
        switch (type) {
        case LogObj::Log2:
            ret = aclnnLog2GetWorkspaceSize(inputTensor, outputTensor,
                                            &workspaceSize, &executor);
            checkASCENDError(ret);

            if (workspaceSize > 0) {
                workspaceAddr = context->getWorkspace(workspaceSize);
            }

            ret = aclnnLog2(workspaceAddr, workspaceSize, executor,
                            context->ASCENDHandle());
            checkASCENDError(ret);
            break;
        case LogObj::LogE:
            ret = aclnnLogGetWorkspaceSize(inputTensor, outputTensor,
                                           &workspaceSize, &executor);
            checkASCENDError(ret);

            if (workspaceSize > 0) {
                workspaceAddr = context->getWorkspace(workspaceSize);
            }

            ret = aclnnLog(workspaceAddr, workspaceSize, executor,
                           context->ASCENDHandle());
            checkASCENDError(ret) break;
        case LogObj::Log10:
            ret = aclnnLog10GetWorkspaceSize(inputTensor, outputTensor,
                                             &workspaceSize, &executor);
            checkASCENDError(ret);

            if (workspaceSize > 0) {
                workspaceAddr = context->getWorkspace(workspaceSize);
            }

            ret = aclnnLog10(workspaceAddr, workspaceSize, executor,
                             context->ASCENDHandle());
            checkASCENDError(ret) break;
        default:
            IT_TODO_HALT();
        }

        aclDestroyTensor(inputTensor);
        aclDestroyTensor(outputTensor);
    }
};

REGISTER_KERNEL(Device::ASCEND, OpType::Log, LogAclnn, "Log_ASCEND");

}; // namespace infini

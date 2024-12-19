#include "aclnnop/level2/aclnn_clamp.h"
#include "ascend/ascend_kernel_without_config.h"
#include "ascend/ascend_runtime.h"
#include "operators/unary.h"

namespace infini {
class ClipAclnn : public ASCENDKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ClipObj>(_op);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        void *min_v = (op->getInputs(1)->getRawDataPtr<void *>());
        void *max_v = (op->getInputs(2)->getRawDataPtr<void *>());

        size_t typeSize = op->getDType().getSize();
        void *min_h = (void *)malloc(typeSize);
        void *max_h = (void *)malloc(typeSize);

        aclrtMemcpy(min_h, typeSize, const_cast<void *>(min_v), typeSize,
                    ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(max_h, typeSize, const_cast<void *>(max_v), typeSize,
                    ACL_MEMCPY_DEVICE_TO_HOST);

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
        aclScalar *max = aclCreateScalar(max_h, aclDataType);
        aclScalar *min = aclCreateScalar(min_h, aclDataType);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;

        auto ret = aclnnClampGetWorkspaceSize(
            inputTensor, min, max, outputTensor, &workspaceSize, &executor);
        checkASCENDError(ret);

        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = context->getWorkspace(workspaceSize);
        }

        ret = aclnnClamp(workspaceAddr, workspaceSize, executor,
                         context->ASCENDHandle());
        checkASCENDError(ret);

        aclDestroyTensor(inputTensor);
        aclDestroyTensor(outputTensor);
        aclDestroyScalar(min);
        aclDestroyScalar(max);
        free(min_h);
        free(max_h);
    }
};

REGISTER_KERNEL(Device::ASCEND, OpType::Clip, ClipAclnn, "Clip_ASCEND");

}; // namespace infini

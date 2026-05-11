#include "operators/unary.h"
#include "aclnnop/level2/aclnn_clamp.h"
#include "ascend/ascend_kernel_without_config.h"
#include "ascend/ascend_runtime.h"
#include "acl/acl.h"

namespace infini {

class ClipAclnn : public ASCENDKernelWithoutConfig {
public:
    void compute(const Operator &_op, const RuntimeObj *_context) const override {
        auto op = as<ClipObj>(_op);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);
        IT_ASSERT(op->getDType() == DataType::Float32);

        // Get input and output pointer
        void *const inputData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const outputData = (op->getOutput()->getRawDataPtr<void *>());

        // Retrieve input and output dimensions and strides
        auto inputDims = op->getInputs(0)->getDims();
        auto inputStride = op->getInputs(0)->getStride();
        auto outputDims = op->getOutput()->getDims();
        auto outputStride = op->getOutput()->getStride();

        // Convert dimensions and strides to int64_t type
        std::vector<int64_t> inputDim64 = castTo64(inputDims);
        std::vector<int64_t> inputStride64 = castTo64(inputStride);
        std::vector<int64_t> outputDim64 = castTo64(outputDims);
        std::vector<int64_t> outputStride64 = castTo64(outputStride);

        // Create ACL tensors
        auto inputTensor = aclCreateTensor(
            inputDim64.data(), inputDim64.size(), ACL_FLOAT, inputStride64.data(), 0,
            aclFormat::ACL_FORMAT_ND, inputDim64.data(), inputDim64.size(), inputData);
        auto outputTensor = aclCreateTensor(
            outputDim64.data(), outputDim64.size(), ACL_FLOAT, outputStride64.data(), 0,
            aclFormat::ACL_FORMAT_ND, outputDim64.data(), outputDim64.size(), outputData);

        // Retrieve the minimum and maximum values for clip
        float clipMin = op->getMin().has_value() 
            ? op->getMin().value() 
            : std::numeric_limits<float>::lowest();
        float clipMax = op->getMax().has_value() 
                    ? op->getMax().value() 
                    : std::numeric_limits<float>::max();

        // Create scalars for the minimum and maximum values
        auto minScalar = aclCreateScalar(&clipMin, ACL_FLOAT);
        auto maxScalar = aclCreateScalar(&clipMax, ACL_FLOAT);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;

        // Retrieve the required workspace size
        auto ret = aclnnClampGetWorkspaceSize(inputTensor, minScalar, maxScalar, outputTensor, &workspaceSize, &executor);
        checkASCENDError(ret);

        // Allocate workspace memory
        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = context->getWorkspace(workspaceSize);
        }

        // Call the Clip computation function
        ret = aclnnClamp(workspaceAddr, workspaceSize, executor, context->ASCENDHandle());
        checkASCENDError(ret);

        // Clean up resources
        aclDestroyTensor(inputTensor);
        aclDestroyTensor(outputTensor);
        aclDestroyScalar(minScalar);
        aclDestroyScalar(maxScalar);

        return;
    }
};

REGISTER_KERNEL(Device::ASCEND, OpType::Clip, ClipAclnn,
                "clip_ASCEND_float");
} // namespace infini
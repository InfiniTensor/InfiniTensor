#include "aclnnop/aclnn_constant_pad_nd.h"
#include "ascend/ascend_kernel_without_config.h"
#include "ascend/ascend_runtime.h"
#include "operators/pad.h"

namespace infini {

class PadAclnn : public ASCENDKernelWithoutConfig {

    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<PadObj>(_op);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);
        IT_ASSERT(op->getDType() == DataType::Float32);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto inputD = op->getInputs(0)->getDims();
        auto inputS = op->getInputs(0)->getStride();

        auto outD = op->getOutput()->getDims();
        auto outS = op->getOutput()->getStride();

        std::vector<int64_t> inputDim = castTo64(inputD);
        std::vector<int64_t> inputStride = castTo64(inputS);

        std::vector<int64_t> outputDim = castTo64(outD);
        std::vector<int64_t> outputStride = castTo64(outS);

        auto inputTensor = aclCreateTensor(
            inputDim.data(), inputDim.size(), ACL_FLOAT, inputStride.data(), 0,
            aclFormat::ACL_FORMAT_ND, inputDim.data(), inputDim.size(), aData);

        auto outputTensor =
            aclCreateTensor(outputDim.data(), outputDim.size(), ACL_FLOAT,
                            outputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                            outputDim.data(), outputDim.size(), cData);

        std::vector<int> intPads = op->getPads();

        std::size_t length = intPads.size();
        std::vector<int64_t> pads(length);
        std::size_t halfLen = length / 2;
        for (std::size_t i = 0; i < halfLen; ++i) {
            pads[2 * i] = intPads[halfLen - i - 1];
            pads[2 * i + 1] = intPads[2 * halfLen - i - 1];
        }

        std::cout << "pads = " << vecToString(pads) << std::endl;

        aclIntArray *padding = aclCreateIntArray(pads.data(), length);
        float valueValue = 0.0f;
        auto value = aclCreateScalar(&valueValue, ACL_FLOAT);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;

        auto ret = aclnnConstantPadNdGetWorkspaceSize(
            inputTensor, padding, value, outputTensor, &workspaceSize,
            &executor);
        checkASCENDError(ret);

        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = context->getWorkspace(workspaceSize);
        }

        ret = aclnnConstantPadNd(workspaceAddr, workspaceSize, executor,
                                 context->ASCENDHandle());
        checkASCENDError(ret);

        aclDestroyTensor(inputTensor);
        aclDestroyIntArray(padding);
        aclDestroyScalar(value);
        aclDestroyTensor(outputTensor);

        return;
    }
};

REGISTER_KERNEL(Device::ASCEND, OpType::Pad, PadAclnn, "pad_ASCEND_float");
}; // namespace infini

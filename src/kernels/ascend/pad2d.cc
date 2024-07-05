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

        std::vector<int> intPads = op->getPads();

        std::size_t length = intPads.size();
        std::vector<int64_t> pads(length);
        std::size_t halfLen = length / 2;
        for (std::size_t i = 0; i < halfLen; ++i) {
            pads[2 * i] = intPads[halfLen - i - 1];
            pads[2 * i + 1] = intPads[2 * halfLen - i - 1];
        }
        // if (length == 8) {
        //     std::size_t halfLen = intPads.size() / 2;
        //     bool condition = true;
        //     // std::cout << "Length of intPads: " << length << std::endl;

        //     // for (std::size_t i = 0; i < halfLen; ++i) {
        //     //     condition = (intPads[i] == intPads[i + 4]);

        //     //     // std::cout << "intPads[" << i << "]: " << intPads[i] <<
        //     //     // std::endl;
        //     // }
        //     assert(condition);

        //     pads[0] = intPads[2];
        //     pads[1] = intPads[6];
        //     pads[2] = intPads[3];
        //     pads[3] = intPads[7];
        // } else if (length == 4) {
        //     for (std::size_t i = 0; i < 4; ++i) {
        //         pads[i] = intPads[i];
        //     }
        // }

        // std::cout << "input shape: " << vecToString(inputDim)
        //           << "  pad output shape: " << vecToString(outD)
        //           << "  pads is: " << vecToString(pads) << std::endl;
        aclIntArray *padding = aclCreateIntArray(pads.data(), length);
        float valueValue = 0.0f;
        auto value = aclCreateScalar(&valueValue, aclDataType);
        auto ret = aclnnConstantPadNdGetWorkspaceSize(
            inputTensor, padding, value, outputTensor, &workspaceSize,
            &executor);
        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = context->getWorkspace(workspaceSize);
        }
        auto tmp_err_msg = aclGetRecentErrMsg();
        if (tmp_err_msg != NULL) {
            printf(" ERROR Message : %s \n ", tmp_err_msg);
        }
        assert(ret == ACL_SUCCESS);
        ret = aclnnConstantPadNd(workspaceAddr, workspaceSize, executor,
                                 context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);

        // aclDestroyTensor(inputTensor);
        // aclDestroyTensor(weightTensor);
        // aclDestroyTensor(outputTensor);

        return;
    }
};

REGISTER_KERNEL(Device::ASCEND, OpType::Pad, PadAclnn, "pad_ASCEND_float");
}; // namespace infini

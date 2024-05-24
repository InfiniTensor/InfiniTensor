#include "aclnnop/level2/aclnn_reflection_pad2d.h"
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

        auto inputTensor =
            aclCreateTensor(inputDim.data(), inputDim.size(), ACL_FLOAT,
                            inputStride.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                            inputDim.data(), inputDim.size(), aData);

        auto outputTensor =
            aclCreateTensor(outputDim.data(), outputDim.size(), ACL_FLOAT,
                            outputStride.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                            outputDim.data(), outputDim.size(), cData);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;

        std::vector<int> intPads = op->getPads();

        std::size_t length = intPads.size();
        std::vector<int64_t> pads(4);
        if (length == 8) {
            std::size_t halfLen = intPads.size() / 2;
            bool condition = true;
            // std::cout << "Length of intPads: " << length << std::endl;

            for (std::size_t i = 0; i < halfLen; ++i) {
                condition = (intPads[i] == intPads[i + 4]);

                // std::cout << "intPads[" << i << "]: " << intPads[i] <<
                // std::endl;
            }
            assert(condition);

            pads[0] = intPads[2];
            pads[1] = intPads[3];
            pads[2] = intPads[6];
            pads[3] = intPads[7];
        } else if (length == 4) {
            for (std::size_t i = 0; i < 4; ++i) {

                pads[i] = intPads[i];
            }
        }

        aclIntArray *padding = aclCreateIntArray(pads.data(), 4);
        auto ret = aclnnReflectionPad2dGetWorkspaceSize(
            inputTensor, padding, outputTensor, &workspaceSize, &executor);
        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = context->getWorkspace(workspaceSize);
        }
        // auto tmp_err_msg = aclGetRecentErrMsg();
        // if (tmp_err_msg != NULL) {
        //     printf(" ERROR Message : %s \n ", tmp_err_msg);
        // }
        assert(ret == ACL_SUCCESS);
        ret = aclnnReflectionPad2d(workspaceAddr, workspaceSize, executor,
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

#include "operators/ascend_dequant.h"
#include "aclnnop/aclnn_ascend_anti_quant.h"
#include "ascend/ascend_kernel_without_config.h"
#include "ascend/ascend_runtime.h"

namespace infini {

class AscendDequantAclnn : public ASCENDKernelWithoutConfig {
    inline int CreateAclTensor(const std::vector<float> &hostData,
                               const std::vector<int64_t> &shape,
                               void **deviceAddr, aclTensor **tensor) const {
        auto size = hostData.size() * sizeof(float);

        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        checkASCENDError(ret);

        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size,
                          ACL_MEMCPY_HOST_TO_DEVICE);
        checkASCENDError(ret);

        // 计算连续tensor的strides
        std::vector<int64_t> strides(1, 1);

        // 调用aclCreateTensor接口创建aclTensor
        *tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT,
                                  strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                  shape.data(), shape.size(), *deviceAddr);
        return 0;
    }

    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<AscendDequantObj>(_op);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);

        void *const inData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const outData = (op->getOutput()->getRawDataPtr<void *>());

        auto inD = op->getInputs(0)->getDims();
        auto inS = op->getInputs(0)->getStride();
        auto outD = op->getOutput()->getDims();
        auto outS = op->getOutput()->getStride();

        std::vector<int64_t> inputDim = castTo64(inD);
        std::vector<int64_t> inputStride = castTo64(inS);
        std::vector<int64_t> outputDim = castTo64(outD);
        std::vector<int64_t> outputStride = castTo64(outS);

        auto inputTensor = aclCreateTensor(
            inputDim.data(), inputDim.size(), ACL_INT8, inputStride.data(), 0,
            aclFormat::ACL_FORMAT_ND, inputDim.data(), inputDim.size(), inData);
        auto outputTensor =
            aclCreateTensor(outputDim.data(), outputDim.size(), ACL_FLOAT16,
                            outputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                            outputDim.data(), outputDim.size(), outData);

        std::vector<float> scalesData = op->getscale();
        std::vector<float> offsetData = op->getoffset();
        std::vector<int64_t> scalesDim(1, scalesData.size());
        std::vector<int64_t> offsetDim(1, offsetData.size());

        void *scaleDeviceAddr = nullptr;
        void *offsetDeviceAddr = nullptr;
        aclTensor *scalesTensor = nullptr;
        aclTensor *offsetTensor = nullptr;

        CreateAclTensor(scalesData, scalesDim, &scaleDeviceAddr, &scalesTensor);
        CreateAclTensor(offsetData, offsetDim, &offsetDeviceAddr,
                        &offsetTensor);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;

        auto ret = aclnnAscendAntiQuantGetWorkspaceSize(
            inputTensor, scalesTensor, offsetTensor, 1, op->getsqrtMode(),
            outputTensor, &workspaceSize, &executor);
        // GetRecentErrMsg();
        checkASCENDError(ret);

        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = context->getWorkspace(workspaceSize);
        }

        ret = aclnnAscendAntiQuant(workspaceAddr, workspaceSize, executor,
                                   context->ASCENDHandle());
        checkASCENDError(ret);

        aclDestroyTensor(inputTensor);
        aclDestroyTensor(outputTensor);
        aclDestroyTensor(scalesTensor);
        aclDestroyTensor(offsetTensor);

        aclrtFree(scaleDeviceAddr);
        aclrtFree(offsetDeviceAddr);

        return;
    }
};

REGISTER_KERNEL(Device::ASCEND, OpType::AscendDequant, AscendDequantAclnn,
                "ascenddequant_ASCEND_float");
} // namespace infini

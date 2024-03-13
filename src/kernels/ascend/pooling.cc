#include "operators/pooling.h"
#include "aclnnop/level2/aclnn_adaptive_max_pool2d.h"
#include "aclnnop/level2/aclnn_avgpool2d.h"
#include "ascend/ascend_kernel_without_config.h"
#include "ascend/ascend_runtime.h"

namespace infini {

class AvgPooling : public ASCENDKernelWithoutConfig {

    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<PoolingObj>(_op);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto [n, c, h, w, kh, kw] = op->getNCHWRS();
        auto [ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();

        std::vector<int64_t> ksize = {kh, kw};
        std::vector<int64_t> stride = {sh, sw};
        std::vector<int64_t> pad = {ph, pw};

        int64_t divisorOverride = 0;

        auto selfD = op->getInputs(0)->getDims();
        auto selfS = op->getInputs(0)->getStride();
        auto outD = op->getOutput()->getDims();
        auto outS = op->getOutput()->getStride();

        std::vector<int64_t> selfDim = castTo64(selfD);
        std::vector<int64_t> selfStride = castTo64(selfS);
        std::vector<int64_t> outputDim = castTo64(outD);
        std::vector<int64_t> outputStride = castTo64(outS);

        aclIntArray *kernelSize = aclCreateIntArray(ksize.data(), ksize.size());
        aclIntArray *strides = aclCreateIntArray(stride.data(), stride.size());
        aclIntArray *paddings = aclCreateIntArray(pad.data(), pad.size());

        auto selfTensor = aclCreateTensor(
            selfDim.data(), selfDim.size(), ACL_FLOAT, selfStride.data(), 0,
            aclFormat::ACL_FORMAT_NCHW, selfDim.data(), selfDim.size(), aData);
        auto outputTensor =
            aclCreateTensor(outputDim.data(), outputDim.size(), ACL_FLOAT,
                            outputStride.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                            outputDim.data(), outputDim.size(), cData);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;

        auto ret = aclnnAvgPool2dGetWorkspaceSize(
            selfTensor, kernelSize, strides, paddings, false, true,
            divisorOverride, 0, outputTensor, &workspaceSize, &executor);
        assert(ret == ACL_SUCCESS);

        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = context->getWorkspace(workspaceSize);
        }

        ret = aclnnAvgPool2d(workspaceAddr, workspaceSize, executor,
                             context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);

        ret = aclrtSynchronizeStream(context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);

        // aclDestroyTensor(selfTensor);
        // aclDestroyTensor(outputTensor);

        return;
    }
};

class MaxPooling : public ASCENDKernelWithoutConfig {
    // Only adaptiveMaxPool2d was found in the ACLNN doc.
    int64_t GetShapeSize(const std::vector<int64_t> &shape) {
        int64_t shapeSize = 1;
        for (auto i : shape) {
            shapeSize *= i;
        }
        return shapeSize;
    }
    template <typename T>
    int CreateAclTensor(const std::vector<T> &hostData,
                        const std::vector<int64_t> &shape, void **deviceAddr,
                        aclDataType dataType, aclTensor **tensor) {
        auto size = GetShapeSize(shape) * sizeof(T);
        // 调用aclrtMalloc申请device侧内存
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        assert(ret == ACL_SUCCESS);
        // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size,
                          ACL_MEMCPY_HOST_TO_DEVICE);
        assert(ret == ACL_SUCCESS);

        // 计算连续tensor的strides
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = shape.size() - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }

        // 调用aclCreateTensor接口创建aclTensor
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                                  strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                                  shape.data(), shape.size(), *deviceAddr);
        return 0;
    }

    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<PoolingObj>(_op);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto selfD = op->getInputs(0)->getDims();
        auto selfS = op->getInputs(0)->getStride();
        auto outD = op->getOutput()->getDims();
        auto outS = op->getOutput()->getStride();

        std::vector<int64_t> selfDim = castTo64(selfD);
        std::vector<int64_t> selfStride = castTo64(selfS);
        std::vector<int64_t> outputDim = castTo64(outD);
        std::vector<int64_t> outputStride = castTo64(outS);

        std::vector<int64_t> outputHW(2, 1);
        outputHW[0] = outputDim[outputDim.size() - 2];
        outputHW[1] = outputDim[outputDim.size() - 1];

        int64_t indicesOutSize = 1;
        for (auto i : outputDim) {
            indicesOutSize *= i;
        }
        void *indicesOutDeviceAddr = nullptr;
        aclrtMalloc(&indicesOutDeviceAddr, indicesOutSize,
                    ACL_MEM_MALLOC_HUGE_FIRST);

        aclIntArray *outputsize =
            aclCreateIntArray(outputHW.data(), outputHW.size());
        auto selfTensor = aclCreateTensor(
            selfDim.data(), selfDim.size(), ACL_FLOAT, selfStride.data(), 0,
            aclFormat::ACL_FORMAT_NCHW, selfDim.data(), selfDim.size(), aData);
        auto outputTensor =
            aclCreateTensor(outputDim.data(), outputDim.size(), ACL_FLOAT,
                            outputStride.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                            outputDim.data(), outputDim.size(), cData);
        auto indicesOutTensor = aclCreateTensor(
            outputDim.data(), outputDim.size(), ACL_INT64, outputStride.data(),
            0, aclFormat::ACL_FORMAT_NCHW, outputDim.data(), outputDim.size(),
            indicesOutDeviceAddr);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;
        auto ret = aclnnAdaptiveMaxPool2dGetWorkspaceSize(
            selfTensor, outputsize, outputTensor, indicesOutTensor,
            &workspaceSize, &executor);
        assert(ret == ACL_SUCCESS);

        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = context->getWorkspace(workspaceSize);
        }

        ret = aclnnAdaptiveMaxPool2d(workspaceAddr, workspaceSize, executor,
                                     context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);

        ret = aclrtSynchronizeStream(context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);

        aclDestroyTensor(indicesOutTensor);

        return;
    }
};

REGISTER_KERNEL(Device::ASCEND, OpType::MaxPool, MaxPooling,
                "maxpooling_ASCEND_float");

REGISTER_KERNEL(Device::ASCEND, OpType::AveragePool, AvgPooling,
                "avgpooling_ASCEND_float");
}; // namespace infini

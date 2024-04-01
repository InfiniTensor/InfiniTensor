#include "operators/pooling.h"
#include "aclnnop/level2/aclnn_avgpool2d.h"
#include "aclnnop/level2/aclnn_max_pool.h"
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
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<PoolingObj>(_op);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto [n, c, h, w, kh, kw] = op->getNCHWRS();
        auto [ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
        int64_t ceilMode = int64_t(op->getCeilMode());

        std::vector<int64_t> ksize = {kh, kw};
        std::vector<int64_t> stride = {sh, sw};
        std::vector<int64_t> pad = {ph, pw};
        std::vector<int64_t> dilation = {dh, dw};

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
        aclIntArray *dilations =
            aclCreateIntArray(dilation.data(), dilation.size());

        auto selfTensor = aclCreateTensor(
            selfDim.data(), selfDim.size(), ACL_FLOAT, selfStride.data(), 0,
            aclFormat::ACL_FORMAT_NCHW, selfDim.data(), selfDim.size(), aData);
        auto outputTensor =
            aclCreateTensor(outputDim.data(), outputDim.size(), ACL_FLOAT,
                            outputStride.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                            outputDim.data(), outputDim.size(), cData);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;
        auto ret = aclnnMaxPoolGetWorkspaceSize(
            selfTensor, kernelSize, strides, 0, paddings, dilations, ceilMode,
            outputTensor, &workspaceSize, &executor);
        assert(ret == ACL_SUCCESS);

        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = context->getWorkspace(workspaceSize);
        }

        ret = aclnnMaxPool(workspaceAddr, workspaceSize, executor,
                           context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);

        ret = aclrtSynchronizeStream(context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);

        return;
    }
};

REGISTER_KERNEL(Device::ASCEND, OpType::MaxPool, MaxPooling,
                "maxpooling_ASCEND_float");

REGISTER_KERNEL(Device::ASCEND, OpType::AveragePool, AvgPooling,
                "avgpooling_ASCEND_float");
}; // namespace infini

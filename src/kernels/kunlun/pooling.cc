#include "operators/pooling.h"
#include "kunlun/kunlun_kernel_without_config.h"
#include "kunlun/kunlun_runtime.h"

namespace infini {
class AvgPooling : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<PoolingObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);
        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto [n, c, h, w, kh, kw] = op->getNCHWRS();
        auto [ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
        auto outShape = op->getOutput()->getDims();

        std::vector<int> ksize = {kh < h ? kh : h, kw < w ? kw : w};
        std::vector<int> stride = {sh, sw};
        std::vector<int> pad = {ph, pw};

        int yh = outShape[op->getOutput()->getRank() - 2];
        int yw = outShape[op->getOutput()->getRank() - 1];

        // If Maxpool with ceilMode true
        // We need to change padding in order to call xdnn api
        if (op->getCeilMode() && yh > (h + 2 * ph - kh) / sh + 1) {
            auto padh = yh - ((h + 2 * ph - kh) / sh + 1);
            auto padw = yw - ((w + 2 * pw - kw) / sw + 1);
            pad = {0, padh, 0, padw};
        }

        auto ret = baidu::xpu::api::avg_pool2d<float>(
            context->KUNLUNHandle(), (float *)aData, (float *)cData, n, c, h, w,
            ksize, stride, pad, true, true, nullptr, nullptr);
        assert(ret == 0);
        return;
    }
};

class MaxPooling : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<PoolingObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);
        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto [n, c, h, w, kh, kw] = op->getNCHWRS();
        auto [ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
        auto outShape = op->getOutput()->getDims();

        std::vector<int> ksize = {kh, kw};
        std::vector<int> stride = {sh, sw};
        std::vector<int> pad = {ph, pw};

        int yh = outShape[op->getOutput()->getRank() - 2];
        int yw = outShape[op->getOutput()->getRank() - 1];

        // If Maxpool with ceilMode true
        // We need to change padding in order to call xdnn api
        if (op->getCeilMode() && yh > (h + 2 * ph - kh) / sh + 1) {
            auto padh = yh - ((h + 2 * ph - kh) / sh + 1);
            auto padw = yw - ((w + 2 * pw - kw) / sw + 1);
            pad = {0, padh, 0, padw};
        }

        KUNLUNPtr indices = context->getWorkspace(yh * yw * sizeof(int));

        checkKUNLUNError(baidu::xpu::api::max_pool2d<float>(
            context->KUNLUNHandle(), (float *)aData, (float *)cData,
            (int *)indices, n, c, h, w, ksize, stride, pad, true, nullptr,
            nullptr, false));

        return;
    }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::MaxPool, MaxPooling, "MaxPool_xdnn");
REGISTER_KERNEL(Device::KUNLUN, OpType::AveragePool, AvgPooling,
                "AvgPool_xdnn");
}; // namespace infini

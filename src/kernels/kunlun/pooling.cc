#include "operators/pooling.h"
#include "kunlun/kunlun_kernel_without_config.h"
#include "kunlun/kunlun_runtime.h"

namespace infini {
class AvgPooling : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<PoolingObj>(_op);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);
        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto [n, c, h, w, kh, kw] = op->getNCHWRS();
        auto [ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();

        std::vector<int> ksize = {kh, kw};
        std::vector<int> stride = {sh, sw};
        std::vector<int> pad = {ph, pw};

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
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);
        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto [n, c, h, w, kh, kw] = op->getNCHWRS();
        auto [ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();

        std::vector<int> ksize = {kh, kw};
        std::vector<int> stride = {sh, sw};
        std::vector<int> pad = {ph, pw};

        int yh = (h + ph * 2 - kh) / sh + 1;
        int yw = (w + pw * 2 - kw) / sw + 1;

        KUNLUNPtr indices = context->getWorkspace(yh * yw * 4);

        auto ret = baidu::xpu::api::max_pool2d<float>(
            context->KUNLUNHandle(), (float *)aData, (float *)cData,
            (int *)indices, n, c, h, w, ksize, stride, pad, true, nullptr,
            nullptr, false);
        assert(ret == 0);
        return;
    }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::MaxPool, DataType::Float32, MaxPooling,
                "MaxPool_xdnn_Float32");
REGISTER_KERNEL(Device::KUNLUN, OpType::AveragePool, DataType::Float32,
                AvgPooling, "AvgPool_xdnn_Float32");
}; // namespace infini

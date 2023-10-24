#include "operators/conv.h"
#include "kunlun/kunlun_kernel_without_config.h"
#include "kunlun/kunlun_runtime.h"

namespace infini {
class ConvXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ConvObj>(_op);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        const auto [ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
        const auto [n, c, h, w, f, r, s] = op->getNCHWFRS();
        const int cpg = op->getChannelPerGroup();
        const int g = c / cpg;

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        std::vector<int> pads = {ph, pw};
        std::vector<int> ksize = {r, s};
        std::vector<int> stride = {sh, sw};
        std::vector<int> dilation = {dh, dw};

        auto ret = baidu::xpu::api::conv2d<float, float, float, float>(
            context->KUNLUNHandle(), (float *)aData, (float *)bData,
            (float *)cData, n, c, h, w, f, ksize, stride, pads, dilation, g,
            nullptr, nullptr, nullptr, true);
        assert(ret == 0);
        return;
    }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::Conv, DataType::Float32, ConvXdnn,
                "Conv_xdnn_KUNLUN_Float32");
}; // namespace infini

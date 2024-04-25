#include "operators/conv.h"
#include "kunlun/kunlun_kernel_without_config.h"
#include "kunlun/kunlun_runtime.h"

namespace infini {
class ConvXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ConvObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
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

        // TODO: Convolution operators still have some accuracy problems
        checkKUNLUNError((xdnn::conv2d<float, float, float, float>(
            context->KUNLUNHandle(), (float *)aData, (float *)bData,
            (float *)cData, n, c, h, w, f, ksize, stride, pads, dilation, g,
            nullptr, nullptr, nullptr, true)));

        // checkKUNLUNError((xdnn::conv2d_fusion<float, float, float, float>(
        //     context->KUNLUNHandle(), (float *const)aData, (float
        //     *const)bData, (float *)cData, n, c, h, w, f, ksize, stride, pads,
        //     dilation, g, nullptr, nullptr, nullptr, true, nullptr, nullptr,
        //     xdnn::Activation_t::LINEAR)));
        return;
    }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::Conv, ConvXdnn, "Conv_xdnn_KUNLUN");
}; // namespace infini

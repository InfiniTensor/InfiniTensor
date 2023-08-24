#include "operators/conv.h"
#include "xpu/xpu_kernel_without_config.h"
#include "xpu/xpu_runtime.h"

namespace infini {
class ConvTransXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ConvBaseObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);

        const auto [ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
        const auto [n, c, h, w, f, r, s] = op->getNCHWFRS();
        const int cpg = op->getChannelPerGroup();
        const int g = c / cpg;
        const bool isNCHW =
            (op->getOpType() == OpType::ConvTransNHWC) ? false : true;

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        std::vector<int> pads = {ph, pw};
        std::vector<int> ksize = {r, s};
        std::vector<int> stride = {sh, sw};
        std::vector<int> dilation = {dh, dw};

        auto dimInputs0 = op->getInputs(0)->getDims();
        auto dimInputs1 = op->getInputs(1)->getDims();
        auto dimOutput = op->getOutput()->getDims();

        if (dimInputs0.size() != 4)
            IT_TODO_HALT();
        if (dimInputs1.size() != 4)
            IT_TODO_HALT();
        if (dimOutput.size() != 4)
            IT_TODO_HALT();

        auto ret =
            baidu::xpu::api::conv2d_transpose<float, float, float, float>(
                context->XPUHandle(), (float *)aData, (float *)bData,
                (float *)cData, n, c, h, w, f, ksize, stride, pads, dilation, g,
                nullptr, nullptr, nullptr, isNCHW);
        assert(ret == 0);
        return;
    }
};

REGISTER_KERNEL(Device::XPU, OpType::ConvTranspose, DataType::Float32,
                ConvTransXdnn, "ConvTrans_xdnn_XPU_Float32");
REGISTER_KERNEL(Device::XPU, OpType::ConvTransNHWC, DataType::Float32,
                ConvTransXdnn, "ConvTranposedNHWC_xdnn_XPU_Float32");

}; // namespace infini

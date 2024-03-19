#include "kunlun/kunlun_kernel_without_config.h"
#include "kunlun/kunlun_runtime.h"
#include "operators/conv.h"

namespace infini {
class ConvTransXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ConvBaseObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

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

        auto ret = xdnn::conv2d_transpose<float, float, float, float>(
            context->KUNLUNHandle(), (float *)aData, (float *)bData,
            (float *)cData, n, c, h, w, f, ksize, stride, pads, dilation, g,
            nullptr, nullptr, nullptr, isNCHW);
        assert(ret == 0);
        return;
    }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::ConvTranspose, ConvTransXdnn,
                "ConvTrans_xdnn_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::ConvTransNHWC, ConvTransXdnn,
                "ConvTranposedNHWC_xdnn_KUNLUN");

}; // namespace infini

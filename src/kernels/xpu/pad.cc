#include "operators/pad.h"
#include "xpu/xpu_kernel_without_config.h"
#include "xpu/xpu_runtime.h"

namespace infini {
class PadXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<PadObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto dim = op->getInputs(0)->getDims();
        int dim_size = dim.size();

        std::vector<int> pads = op->getPads();

        std::cout << std::endl;
        std::vector<int> paddings_left(pads.begin(), pads.begin() + dim_size);
        std::vector<int> paddings_right(pads.begin() + dim_size, pads.end());

        float paddingValue = 0.0;
        auto ret = baidu::xpu::api::pad<float>(
            context->XPUHandle(), (float *)aData, (float *)cData, dim,
            paddings_left, paddings_right, paddingValue);

        assert(ret == 0);
        return;
    }
};

REGISTER_KERNEL(Device::XPU, OpType::Pad, DataType::Float32, PadXdnn,
                "Pad_xdnn_XPU_Float32");

}; // namespace infini

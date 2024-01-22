#include "operators/pad.h"
#include "kunlun/kunlun_kernel_without_config.h"
#include "kunlun/kunlun_runtime.h"

namespace infini {
class PadXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<PadObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

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
            context->KUNLUNHandle(), (float *)aData, (float *)cData, dim,
            paddings_left, paddings_right, paddingValue);

        assert(ret == 0);
        return;
    }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::Pad, PadXdnn, "Pad_xdnn_KUNLUN");

}; // namespace infini

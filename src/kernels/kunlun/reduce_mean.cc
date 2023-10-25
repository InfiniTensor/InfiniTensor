#include "operators/reduce_mean.h"
#include "kunlun/kunlun_kernel_without_config.h"
#include "kunlun/kunlun_runtime.h"

namespace infini {
class ReduceMeanXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ReduceMeanObj>(_op);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

	auto axes_set = op->getAxes();
        std::vector<int> axes;
        axes.assign(axes_set.begin(), axes_set.end());
	auto shape = op->getInputs(0)->getDims();

        auto ret = baidu::xpu::api::reduce_mean<float>(
            context->KUNLUNHandle(), (float *)aData, (float *)cData, shape, axes);
        assert(ret == 0);
        return;
    }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::ReduceMean, DataType::Float32, ReduceMeanXdnn,
                "ReduceMean_xdnn_KUNLUN_Float32");
}; // namespace infini

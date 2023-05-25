#include "operators/element_wise.h"
#include "xpu/xpu_kernel_without_config.h"
#include "xpu/xpu_runtime.h"

namespace infini {
class AddXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto aDim = op->getInputs(0)->getDims();
        auto bDim = op->getInputs(1)->getDims();
        if (aDim.size() != 4 || bDim.size() != 4)
            IT_TODO_HALT();
	auto ret = baidu::xpu::api::broadcast_add<float>(context->XPUHandle(), (float*)aData, (float*)bData, (float*)cData, aDim, bDim);
	assert(ret == 0);
	return;

    }
};

REGISTER_KERNEL(Device::XPU, OpType::Add, DataType::Float32, AddXdnn,
                "Add_xdnn_XPU_Float32");
}; // namespace infini

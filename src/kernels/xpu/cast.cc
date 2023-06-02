#include "operators/unary.h"
#include "xpu/xpu_kernel_without_config.h"
#include "xpu/xpu_runtime.h"

namespace infini {
class CastXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<CastObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();
	CastObj::CastType type = op->getType();

	int ret = 0;
	switch (type) {
		case CastObj::Float2Int32:
			ret = baidu::xpu::api::cast<float,int>(context->XPUHandle(), (float*)aData, (int*)cData, len);
			break;
		case CastObj::Int322Int8:
			ret = baidu::xpu::api::cast<int,float>(context->XPUHandle(), (int*)aData, (float*)cData, len);
			break;
		default:
			IT_TODO_HALT();

	} 
	assert(ret == 0);
	return;

    }
};

REGISTER_KERNEL(Device::XPU, OpType::Cast, DataType::Float32, CastXdnn,
                "Cast_xdnn_XPU_Float32");
}; // namespace infini

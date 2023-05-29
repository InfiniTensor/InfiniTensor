#include "operators/concat.h"
#include "xpu/xpu_kernel_without_config.h"
#include "xpu/xpu_runtime.h"

namespace infini {
class ConcatXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ConcatObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);
	int axis = op->getDim();
	int num = op->numInputs();
	std::vector<const float*> inputsData;
	for (int i = 0; i < num; ++i) {
		inputsData.push_back((float*)(op->getInputs(i)->getRawDataPtr<void *>()));
	}
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

	std::vector<std::vector<int>> dims;
	for(int i = 0; i < num; ++i){
        	auto dim = op->getInputs(i)->getDims();
        	if (dim.size() != 4) {
			IT_TODO_HALT();
		}
		dims.push_back(dim);
	}
	auto ret = baidu::xpu::api::concat<float>(context->XPUHandle(), inputsData, (float*)cData, dims, axis);
	assert(ret == 0);
	return;

    }
};

REGISTER_KERNEL(Device::XPU, OpType::Concat, DataType::Float32, ConcatXdnn,
                "Concat_xdnn_XPU_Float32");
}; // namespace infini

#include "operators/unary.h"
#include "xpu/xpu_kernel_without_config.h"
#include "xpu/xpu_runtime.h"

namespace infini {
class ReluXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();

	auto ret = baidu::xpu::api::relu<float>(context->XPUHandle(), (float*)aData, (float*)cData, len);
	assert(ret == 0);
	return;

    }
};

class SigmoidXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();

	auto ret = baidu::xpu::api::sigmoid<float>(context->XPUHandle(), (float*)aData, (float*)cData, len);
	assert(ret == 0);
	return;

    }
};

class TanhXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();

	auto ret = baidu::xpu::api::tanh<float>(context->XPUHandle(), (float*)aData, (float*)cData, len);
	assert(ret == 0);
	return;

    }
};

class SquareXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();

	auto ret = baidu::xpu::api::square<float>(context->XPUHandle(), (float*)aData, (float*)cData, len);
	assert(ret == 0);
	return;

    }
};


REGISTER_KERNEL(Device::XPU, OpType::Relu, DataType::Float32, ReluXdnn,
                "Relu_xdnn_XPU_Float32");
REGISTER_KERNEL(Device::XPU, OpType::Sigmoid, DataType::Float32, SigmoidXdnn,
                "Sigmoid_xdnn_XPU_Float32");
REGISTER_KERNEL(Device::XPU, OpType::Tanh, DataType::Float32, TanhXdnn,
                "Tanh_xdnn_XPU_Float32");
REGISTER_KERNEL(Device::XPU, OpType::Square, DataType::Float32, SquareXdnn,
                "Square_xdnn_XPU_Float32");
}; // namespace infini

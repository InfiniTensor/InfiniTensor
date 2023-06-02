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

class SqrtXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();

	auto ret = baidu::xpu::api::sqrt<float>(context->XPUHandle(), (float*)aData, (float*)cData, len);
	assert(ret == 0);
	return;

    }
};

class RsqrtXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();

	auto ret = baidu::xpu::api::rsqrt<float>(context->XPUHandle(), (float*)aData, (float*)cData, len);
	assert(ret == 0);
	return;

    }
};

class ExpXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();

	auto ret = baidu::xpu::api::exp<float>(context->XPUHandle(), (float*)aData, (float*)cData, len);
	assert(ret == 0);
	return;

    }
};

class CeilXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();

	auto ret = baidu::xpu::api::ceil<float>(context->XPUHandle(), (float*)aData, (float*)cData, len);
	assert(ret == 0);
	return;

    }
};

class ClipXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ClipObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();
	float min = op->getMin().value();
	float max = op->getMax().value();

	auto ret = baidu::xpu::api::clip<float>(context->XPUHandle(), (float*)aData, (float*)cData, len, min, max);
	assert(ret == 0);
	return;

    }
};

class FloorXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();

	auto ret = baidu::xpu::api::floor<float>(context->XPUHandle(), (float*)aData, (float*)cData, len);
	assert(ret == 0);
	return;

    }
};

class NegXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();

	auto ret = baidu::xpu::api::neg<float>(context->XPUHandle(), (float*)aData, (float*)cData, len);
	assert(ret == 0);
	return;

    }
};

class CopyXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();

	auto ret = baidu::xpu::api::copy<float>(context->XPUHandle(), (float*)aData, (float*)cData, len);
	assert(ret == 0);
	return;

    }
};

class ReciprocalXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();

	auto ret = baidu::xpu::api::reciprocal<float>(context->XPUHandle(), (float*)aData, (float*)cData, len);
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
REGISTER_KERNEL(Device::XPU, OpType::Sqrt, DataType::Float32, SqrtXdnn,
                "Sqrt_xdnn_XPU_Float32");
REGISTER_KERNEL(Device::XPU, OpType::Rsqrt, DataType::Float32, RsqrtXdnn,
                "Rsqrt_xdnn_XPU_Float32");
REGISTER_KERNEL(Device::XPU, OpType::Exp, DataType::Float32, ExpXdnn,
                "Exp_xdnn_XPU_Float32");
REGISTER_KERNEL(Device::XPU, OpType::Ceil, DataType::Float32, CeilXdnn,
                "Ceil_xdnn_XPU_Float32");
REGISTER_KERNEL(Device::XPU, OpType::Clip, DataType::Float32, ClipXdnn,
                "Clip_xdnn_XPU_Float32");
REGISTER_KERNEL(Device::XPU, OpType::Floor, DataType::Float32, FloorXdnn,
                "Floor_xdnn_XPU_Float32");
REGISTER_KERNEL(Device::XPU, OpType::Neg, DataType::Float32, NegXdnn,
                "Neg_xdnn_XPU_Float32");
REGISTER_KERNEL(Device::XPU, OpType::Copy, DataType::Float32, CopyXdnn,
                "Copy_xdnn_XPU_Float32");
REGISTER_KERNEL(Device::XPU, OpType::Reciprocal, DataType::Float32, ReciprocalXdnn,
                "Reciprocal_xdnn_XPU_Float32");
}; // namespace infini

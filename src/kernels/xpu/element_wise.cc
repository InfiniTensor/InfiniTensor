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

class SubXdnn : public XPUKernelWithoutConfig {
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
	auto ret = baidu::xpu::api::broadcast_sub<float>(context->XPUHandle(), (float*)aData, (float*)bData, (float*)cData, aDim, bDim);
	assert(ret == 0);
	return;

    }
};

class MulXdnn : public XPUKernelWithoutConfig {
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
	auto ret = baidu::xpu::api::broadcast_mul<float>(context->XPUHandle(), (float*)aData, (float*)bData, (float*)cData, aDim, bDim);
	assert(ret == 0);
	return;

    }
};

class DivXdnn : public XPUKernelWithoutConfig {
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
	auto ret = baidu::xpu::api::broadcast_div<float>(context->XPUHandle(), (float*)aData, (float*)bData, (float*)cData, aDim, bDim);
	assert(ret == 0);
	return;

    }
};

class PowXdnn : public XPUKernelWithoutConfig {
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
	auto ret = baidu::xpu::api::broadcast_pow<float>(context->XPUHandle(), (float*)aData, (float*)bData, (float*)cData, aDim, bDim);
	assert(ret == 0);
	return;

    }
};

class MaxXdnn : public XPUKernelWithoutConfig {
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
	auto ret = baidu::xpu::api::broadcast_max<float>(context->XPUHandle(), (float*)aData, (float*)bData, (float*)cData, aDim, bDim);
	assert(ret == 0);
	return;

    }
};

class MinXdnn : public XPUKernelWithoutConfig {
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
	auto ret = baidu::xpu::api::broadcast_min<float>(context->XPUHandle(), (float*)aData, (float*)bData, (float*)cData, aDim, bDim);
	assert(ret == 0);
	return;

    }
};

REGISTER_KERNEL(Device::XPU, OpType::Add, DataType::Float32, AddXdnn,
                "Add_xdnn_XPU_Float32");
REGISTER_KERNEL(Device::XPU, OpType::Sub, DataType::Float32, SubXdnn,
                "Sub_xdnn_XPU_Float32");
REGISTER_KERNEL(Device::XPU, OpType::Mul, DataType::Float32, MulXdnn,
                "Mul_xdnn_XPU_Float32");
REGISTER_KERNEL(Device::XPU, OpType::Div, DataType::Float32, DivXdnn,
                "Div_xdnn_XPU_Float32");
REGISTER_KERNEL(Device::XPU, OpType::Pow, DataType::Float32, PowXdnn,
                "Pow_xdnn_XPU_Float32");
REGISTER_KERNEL(Device::XPU, OpType::Maximum, DataType::Float32, MaxXdnn,
                "Max_xdnn_XPU_Float32");
REGISTER_KERNEL(Device::XPU, OpType::Minimum, DataType::Float32, MinXdnn,
                "Min_xdnn_XPU_Float32");
}; // namespace infini

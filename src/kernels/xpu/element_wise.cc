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
        auto ret = baidu::xpu::api::broadcast_add<float>(
            context->XPUHandle(), (float *)aData, (float *)bData,
            (float *)cData, aDim, bDim);
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
        auto ret = baidu::xpu::api::broadcast_sub<float>(
            context->XPUHandle(), (float *)aData, (float *)bData,
            (float *)cData, aDim, bDim);
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
        auto ret = baidu::xpu::api::broadcast_mul<float>(
            context->XPUHandle(), (float *)aData, (float *)bData,
            (float *)cData, aDim, bDim);
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
        auto ret = baidu::xpu::api::broadcast_div<float>(
            context->XPUHandle(), (float *)aData, (float *)bData,
            (float *)cData, aDim, bDim);
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
        auto ret = baidu::xpu::api::broadcast_pow<float>(
            context->XPUHandle(), (float *)aData, (float *)bData,
            (float *)cData, aDim, bDim);
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
        auto ret = baidu::xpu::api::broadcast_max<float>(
            context->XPUHandle(), (float *)aData, (float *)bData,
            (float *)cData, aDim, bDim);
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
        auto ret = baidu::xpu::api::broadcast_min<float>(
            context->XPUHandle(), (float *)aData, (float *)bData,
            (float *)cData, aDim, bDim);
        assert(ret == 0);
        return;
    }
};

class EqualXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        size_t len = op->getOutput()->size();
        XPUPtr wsData = context->getWorkspace(len);

        auto aDim = op->getInputs(0)->getDims();
        auto bDim = op->getInputs(1)->getDims();
        if (aDim.size() != 4 || bDim.size() != 4)
            IT_TODO_HALT();
        auto ret = baidu::xpu::api::broadcast_equal<float>(
            context->XPUHandle(), (float *)aData, (float *)bData,
            (bool *)wsData, aDim, bDim);
        ret = baidu::xpu::api::cast<bool, float>(
            context->XPUHandle(), (bool *)wsData, (float *)cData, len);
        assert(ret == 0);
        return;
    }
};

class GreaterEqualXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        size_t len = op->getOutput()->size();
        XPUPtr wsData = context->getWorkspace(len);

        auto aDim = op->getInputs(0)->getDims();
        auto bDim = op->getInputs(1)->getDims();
        if (aDim.size() != 4 || bDim.size() != 4)
            IT_TODO_HALT();
        auto ret = baidu::xpu::api::broadcast_greater_equal<float>(
            context->XPUHandle(), (float *)aData, (float *)bData,
            (bool *)wsData, aDim, bDim);
        ret = baidu::xpu::api::cast<bool, float>(
            context->XPUHandle(), (bool *)wsData, (float *)cData, len);
        assert(ret == 0);
        return;
    }
};

class GreaterThanXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        size_t len = op->getOutput()->size();
        XPUPtr wsData = context->getWorkspace(len);

        auto aDim = op->getInputs(0)->getDims();
        auto bDim = op->getInputs(1)->getDims();
        if (aDim.size() != 4 || bDim.size() != 4)
            IT_TODO_HALT();
        auto ret = baidu::xpu::api::broadcast_greater_than<float>(
            context->XPUHandle(), (float *)aData, (float *)bData,
            (bool *)wsData, aDim, bDim);
        ret = baidu::xpu::api::cast<bool, float>(
            context->XPUHandle(), (bool *)wsData, (float *)cData, len);
        assert(ret == 0);
        return;
    }
};

class LessEqualXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        size_t len = op->getOutput()->size();
        XPUPtr wsData = context->getWorkspace(len);

        auto aDim = op->getInputs(0)->getDims();
        auto bDim = op->getInputs(1)->getDims();
        if (aDim.size() != 4 || bDim.size() != 4)
            IT_TODO_HALT();
        auto ret = baidu::xpu::api::broadcast_less_equal<float>(
            context->XPUHandle(), (float *)aData, (float *)bData,
            (bool *)wsData, aDim, bDim);
        ret = baidu::xpu::api::cast<bool, float>(
            context->XPUHandle(), (bool *)wsData, (float *)cData, len);
        assert(ret == 0);
        return;
    }
};

class LessThanXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        size_t len = op->getOutput()->size();
        XPUPtr wsData = context->getWorkspace(len);

        auto aDim = op->getInputs(0)->getDims();
        auto bDim = op->getInputs(1)->getDims();
        if (aDim.size() != 4 || bDim.size() != 4)
            IT_TODO_HALT();
        auto ret = baidu::xpu::api::broadcast_less_than<float>(
            context->XPUHandle(), (float *)aData, (float *)bData,
            (bool *)wsData, aDim, bDim);
        ret = baidu::xpu::api::cast<bool, float>(
            context->XPUHandle(), (bool *)wsData, (float *)cData, len);
        assert(ret == 0);
        return;
    }
};

class FloorDivXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        size_t len = op->getOutput()->size();
        XPUPtr wsData = context->getWorkspace(len);

        auto aDim = op->getInputs(0)->getDims();
        auto bDim = op->getInputs(1)->getDims();
        if (aDim.size() != 4 || bDim.size() != 4)
            IT_TODO_HALT();
        auto ret = baidu::xpu::api::broadcast_floordiv<float>(
            context->XPUHandle(), (float *)aData, (float *)bData,
            (float *)wsData, aDim, bDim);
        ret = baidu::xpu::api::cast<int, float>(
            context->XPUHandle(), (int *)wsData, (float *)cData, len);
        assert(ret == 0);
        return;
    }
};

class MSELossXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<MSELossObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        size_t len = op->getOutput()->size();

        auto dim = op->getInputs(0)->getDims();
        if (dim.size() != 4)
            IT_TODO_HALT();

        auto ret = baidu::xpu::api::mse_loss<float>(
            context->XPUHandle(), (float *)aData, (float *)bData,
            (float *)cData, len);
        assert(ret == 0);
        return;
    }
};

class AndXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        size_t len = op->getOutput()->size();
        XPUPtr wsData = context->getWorkspace(len);

        auto aDim = op->getInputs(0)->getDims();
        auto bDim = op->getInputs(1)->getDims();
        if (aDim.size() != 4 || bDim.size() != 4)
            IT_TODO_HALT();
        auto ret = baidu::xpu::api::logical_and<bool>(
            context->XPUHandle(), (bool *)aData, (bool *)bData, (bool *)wsData,
            len);
        ret = baidu::xpu::api::cast<bool, float>(
            context->XPUHandle(), (bool *)wsData, (float *)cData, len);
        assert(ret == 0);
        return;
    }
};

class OrXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        size_t len = op->getOutput()->size();
        XPUPtr wsData = context->getWorkspace(len);

        auto aDim = op->getInputs(0)->getDims();
        auto bDim = op->getInputs(1)->getDims();
        if (aDim.size() != 4 || bDim.size() != 4)
            IT_TODO_HALT();
        auto ret = baidu::xpu::api::logical_or<bool>(
            context->XPUHandle(), (bool *)aData, (bool *)bData, (bool *)wsData,
            len);
        ret = baidu::xpu::api::cast<bool, float>(
            context->XPUHandle(), (bool *)wsData, (float *)cData, len);
        assert(ret == 0);
        return;
    }
};

class XorXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        size_t len = op->getOutput()->size();
        XPUPtr wsData = context->getWorkspace(len);

        auto aDim = op->getInputs(0)->getDims();
        auto bDim = op->getInputs(1)->getDims();
        if (aDim.size() != 4 || bDim.size() != 4)
            IT_TODO_HALT();
        auto ret = baidu::xpu::api::logical_xor<bool>(
            context->XPUHandle(), (bool *)aData, (bool *)bData, (bool *)wsData,
            len);
        ret = baidu::xpu::api::cast<bool, float>(
            context->XPUHandle(), (bool *)wsData, (float *)cData, len);
        assert(ret == 0);
        return;
    }
};

class NotXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        size_t len = op->getOutput()->size();
        XPUPtr wsData = context->getWorkspace(len);

        auto aDim = op->getInputs(0)->getDims();
        if (aDim.size() != 4)
            IT_TODO_HALT();
        auto ret = baidu::xpu::api::logical_not<bool>(
            context->XPUHandle(), (bool *)aData, (bool *)wsData, len);
        ret = baidu::xpu::api::cast<bool, float>(
            context->XPUHandle(), (bool *)wsData, (float *)cData, len);
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
REGISTER_KERNEL(Device::XPU, OpType::Max, DataType::Float32, MaxXdnn,
                "Max_xdnn_XPU_Float32");
REGISTER_KERNEL(Device::XPU, OpType::Min, DataType::Float32, MinXdnn,
                "Min_xdnn_XPU_Float32");
REGISTER_KERNEL(Device::XPU, OpType::Equal, DataType::Float32, EqualXdnn,
                "Equal_xdnn_XPU_Float32");
REGISTER_KERNEL(Device::XPU, OpType::GreaterOrEqual, DataType::Float32,
                GreaterEqualXdnn, "GreaterEqual_xdnn_XPU_Float32");
REGISTER_KERNEL(Device::XPU, OpType::Greater, DataType::Float32,
                GreaterThanXdnn, "GreaterThan_xdnn_XPU_Float32");
REGISTER_KERNEL(Device::XPU, OpType::LessOrEqual, DataType::Float32,
                LessEqualXdnn, "LessEqual_xdnn_XPU_Float32");
REGISTER_KERNEL(Device::XPU, OpType::Less, DataType::Float32, LessThanXdnn,
                "LessThan_xdnn_XPU_Float32");
REGISTER_KERNEL(Device::XPU, OpType::FloorDiv, DataType::Float32, FloorDivXdnn,
                "FloorDiv_xdnn_XPU_Float32");
REGISTER_KERNEL(Device::XPU, OpType::MSELoss, DataType::Float32, MSELossXdnn,
                "MSELoss_xdnn_XPU_Float32");
REGISTER_KERNEL(Device::XPU, OpType::And, DataType::Float32, AndXdnn,
                "And_xdnn_XPU_Float32");
REGISTER_KERNEL(Device::XPU, OpType::Or, DataType::Float32, OrXdnn,
                "Or_xdnn_XPU_Float32");
REGISTER_KERNEL(Device::XPU, OpType::Xor, DataType::Float32, XorXdnn,
                "Xor_xdnn_XPU_Float32");
REGISTER_KERNEL(Device::XPU, OpType::Not, DataType::Float32, NotXdnn,
                "Not_xdnn_XPU_Float32");
}; // namespace infini

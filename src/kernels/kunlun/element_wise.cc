#include "operators/element_wise.h"
#include "kunlun/kunlun_kernel_without_config.h"
#include "kunlun/kunlun_runtime.h"

namespace infini {
class AddXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto aDim = op->getInputs(0)->getDims();
        auto bDim = op->getInputs(1)->getDims();
        if (aDim.size() == 0) {
            aDim.push_back(1);
        }
        if (bDim.size() == 0) {
            bDim.push_back(1);
        }
        auto ret = baidu::xpu::api::broadcast_add<float>(
            context->KUNLUNHandle(), (float *)aData, (float *)bData,
            (float *)cData, aDim, bDim);
        assert(ret == 0);
        return;
    }
};

class SubXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto aDim = op->getInputs(0)->getDims();
        auto bDim = op->getInputs(1)->getDims();
        if (aDim.size() == 0) {
            aDim.push_back(1);
        }
        if (bDim.size() == 0) {
            bDim.push_back(1);
        }
        auto ret = baidu::xpu::api::broadcast_sub<float>(
            context->KUNLUNHandle(), (float *)aData, (float *)bData,
            (float *)cData, aDim, bDim);
        assert(ret == 0);
        return;
    }
};

class MulXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto aDim = op->getInputs(0)->getDims();
        auto bDim = op->getInputs(1)->getDims();
        if (aDim.size() == 0) {
            aDim.push_back(1);
        }
        if (bDim.size() == 0) {
            bDim.push_back(1);
        }
        auto ret = baidu::xpu::api::broadcast_mul<float>(
            context->KUNLUNHandle(), (float *)aData, (float *)bData,
            (float *)cData, aDim, bDim);
        assert(ret == 0);
        return;
    }
};

class DivXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto aDim = op->getInputs(0)->getDims();
        auto bDim = op->getInputs(1)->getDims();
        if (aDim.size() == 0) {
            aDim.push_back(1);
        }
        if (bDim.size() == 0) {
            bDim.push_back(1);
        }
        auto ret = baidu::xpu::api::broadcast_div<float>(
            context->KUNLUNHandle(), (float *)aData, (float *)bData,
            (float *)cData, aDim, bDim);
        assert(ret == 0);
        return;
    }
};

class PowXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto aDim = op->getInputs(0)->getDims();
        auto bDim = op->getInputs(1)->getDims();
        if (aDim.size() == 0) {
            aDim.push_back(1);
        }
        if (bDim.size() == 0) {
            bDim.push_back(1);
        }

        auto ret = baidu::xpu::api::broadcast_pow<float>(
            context->KUNLUNHandle(), (float *)aData, (float *)bData,
            (float *)cData, aDim, bDim);
        assert(ret == 0);
        return;
    }
};

class MaxXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto aDim = op->getInputs(0)->getDims();
        auto bDim = op->getInputs(1)->getDims();
        if (aDim.size() == 0) {
            aDim.push_back(1);
        }
        if (bDim.size() == 0) {
            bDim.push_back(1);
        }
        auto ret = baidu::xpu::api::broadcast_max<float>(
            context->KUNLUNHandle(), (float *)aData, (float *)bData,
            (float *)cData, aDim, bDim);
        assert(ret == 0);
        return;
    }
};

class MinXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto aDim = op->getInputs(0)->getDims();
        auto bDim = op->getInputs(1)->getDims();
        if (aDim.size() == 0) {
            aDim.push_back(1);
        }
        if (bDim.size() == 0) {
            bDim.push_back(1);
        }
        auto ret = baidu::xpu::api::broadcast_min<float>(
            context->KUNLUNHandle(), (float *)aData, (float *)bData,
            (float *)cData, aDim, bDim);
        assert(ret == 0);
        return;
    }
};

class EqualXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        size_t len = op->getOutput()->size();
        KUNLUNPtr wsData = context->getWorkspace(len);

        auto aDim = op->getInputs(0)->getDims();
        auto bDim = op->getInputs(1)->getDims();
        if (aDim.size() == 0) {
            aDim.push_back(1);
        }
        if (bDim.size() == 0) {
            bDim.push_back(1);
        }
        auto ret = baidu::xpu::api::broadcast_equal<float>(
            context->KUNLUNHandle(), (float *)aData, (float *)bData,
            (bool *)wsData, aDim, bDim);
        ret = baidu::xpu::api::cast<bool, float>(
            context->KUNLUNHandle(), (bool *)wsData, (float *)cData, len);
        assert(ret == 0);
        return;
    }
};

class GreaterEqualXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        size_t len = op->getOutput()->size();
        KUNLUNPtr wsData = context->getWorkspace(len);

        auto aDim = op->getInputs(0)->getDims();
        auto bDim = op->getInputs(1)->getDims();
        if (aDim.size() == 0) {
            aDim.push_back(1);
        }
        if (bDim.size() == 0) {
            bDim.push_back(1);
        }
        auto ret = baidu::xpu::api::broadcast_greater_equal<float>(
            context->KUNLUNHandle(), (float *)aData, (float *)bData,
            (bool *)wsData, aDim, bDim);
        ret = baidu::xpu::api::cast<bool, float>(
            context->KUNLUNHandle(), (bool *)wsData, (float *)cData, len);
        assert(ret == 0);
        return;
    }
};

class GreaterThanXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        size_t len = op->getOutput()->size();
        KUNLUNPtr wsData = context->getWorkspace(len);

        auto aDim = op->getInputs(0)->getDims();
        auto bDim = op->getInputs(1)->getDims();
        if (aDim.size() == 0) {
            aDim.push_back(1);
        }
        if (bDim.size() == 0) {
            bDim.push_back(1);
        }
        auto ret = baidu::xpu::api::broadcast_greater_than<float>(
            context->KUNLUNHandle(), (float *)aData, (float *)bData,
            (bool *)wsData, aDim, bDim);
        ret = baidu::xpu::api::cast<bool, float>(
            context->KUNLUNHandle(), (bool *)wsData, (float *)cData, len);
        assert(ret == 0);
        return;
    }
};

class LessEqualXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        size_t len = op->getOutput()->size();
        KUNLUNPtr wsData = context->getWorkspace(len);

        auto aDim = op->getInputs(0)->getDims();
        auto bDim = op->getInputs(1)->getDims();
        if (aDim.size() == 0) {
            aDim.push_back(1);
        }
        if (bDim.size() == 0) {
            bDim.push_back(1);
        }
        auto ret = baidu::xpu::api::broadcast_less_equal<float>(
            context->KUNLUNHandle(), (float *)aData, (float *)bData,
            (bool *)wsData, aDim, bDim);
        ret = baidu::xpu::api::cast<bool, float>(
            context->KUNLUNHandle(), (bool *)wsData, (float *)cData, len);
        assert(ret == 0);
        return;
    }
};

class LessThanXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        size_t len = op->getOutput()->size();
        KUNLUNPtr wsData = context->getWorkspace(len);

        auto aDim = op->getInputs(0)->getDims();
        auto bDim = op->getInputs(1)->getDims();
        if (aDim.size() == 0) {
            aDim.push_back(1);
        }
        if (bDim.size() == 0) {
            bDim.push_back(1);
        }
        auto ret = baidu::xpu::api::broadcast_less_than<float>(
            context->KUNLUNHandle(), (float *)aData, (float *)bData,
            (bool *)wsData, aDim, bDim);
        ret = baidu::xpu::api::cast<bool, float>(
            context->KUNLUNHandle(), (bool *)wsData, (float *)cData, len);
        assert(ret == 0);
        return;
    }
};

class FloorDivXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto aDim = op->getInputs(0)->getDims();
        auto bDim = op->getInputs(1)->getDims();
        if (aDim.size() == 0) {
            aDim.push_back(1);
        }
        if (bDim.size() == 0) {
            bDim.push_back(1);
        }
        auto ret = baidu::xpu::api::broadcast_floordiv<float>(
            context->KUNLUNHandle(), (float *)aData, (float *)bData,
            (float *)cData, aDim, bDim);
        assert(ret == 0);
        return;
    }
};

class MSELossXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<MSELossObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        size_t len = op->getOutput()->size();

        auto dim = op->getInputs(0)->getDims();
        auto ret = baidu::xpu::api::mse_loss<float>(
            context->KUNLUNHandle(), (float *)aData, (float *)bData,
            (float *)cData, len);
        assert(ret == 0);
        return;
    }
};

class AndXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        size_t len = op->getOutput()->size();
        KUNLUNPtr wsData = context->getWorkspace(len);

        auto aDim = op->getInputs(0)->getDims();
        auto bDim = op->getInputs(1)->getDims();
        if (aDim.size() == 0) {
            aDim.push_back(1);
        }
        if (bDim.size() == 0) {
            bDim.push_back(1);
        }
        auto ret = baidu::xpu::api::logical_and<bool>(
            context->KUNLUNHandle(), (bool *)aData, (bool *)bData,
            (bool *)wsData, len);
        ret = baidu::xpu::api::cast<bool, float>(
            context->KUNLUNHandle(), (bool *)wsData, (float *)cData, len);
        assert(ret == 0);
        return;
    }
};

class OrXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        size_t len = op->getOutput()->size();
        KUNLUNPtr wsData = context->getWorkspace(len);

        auto aDim = op->getInputs(0)->getDims();
        auto bDim = op->getInputs(1)->getDims();
        if (aDim.size() == 0) {
            aDim.push_back(1);
        }
        if (bDim.size() == 0) {
            bDim.push_back(1);
        }
        auto ret = baidu::xpu::api::logical_or<bool>(
            context->KUNLUNHandle(), (bool *)aData, (bool *)bData,
            (bool *)wsData, len);
        ret = baidu::xpu::api::cast<bool, float>(
            context->KUNLUNHandle(), (bool *)wsData, (float *)cData, len);
        assert(ret == 0);
        return;
    }
};

class XorXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        size_t len = op->getOutput()->size();
        KUNLUNPtr wsData = context->getWorkspace(len);

        auto aDim = op->getInputs(0)->getDims();
        auto bDim = op->getInputs(1)->getDims();
        if (aDim.size() == 0) {
            aDim.push_back(1);
        }
        if (bDim.size() == 0) {
            bDim.push_back(1);
        }
        auto ret = baidu::xpu::api::logical_xor<bool>(
            context->KUNLUNHandle(), (bool *)aData, (bool *)bData,
            (bool *)wsData, len);
        ret = baidu::xpu::api::cast<bool, float>(
            context->KUNLUNHandle(), (bool *)wsData, (float *)cData, len);
        assert(ret == 0);
        return;
    }
};

class NotXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        size_t len = op->getOutput()->size();
        KUNLUNPtr wsData = context->getWorkspace(len);

        auto aDim = op->getInputs(0)->getDims();
        auto ret = baidu::xpu::api::logical_not<bool>(
            context->KUNLUNHandle(), (bool *)aData, (bool *)wsData, len);
        ret = baidu::xpu::api::cast<bool, float>(
            context->KUNLUNHandle(), (bool *)wsData, (float *)cData, len);
        assert(ret == 0);
        return;
    }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::Add, AddXdnn, "Add_xdnn_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::Sub, SubXdnn, "Sub_xdnn_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::Mul, MulXdnn, "Mul_xdnn_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::Div, DivXdnn, "Div_xdnn_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::Pow, PowXdnn, "Pow_xdnn_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::Max, MaxXdnn, "Max_xdnn_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::Min, MinXdnn, "Min_xdnn_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::Equal, EqualXdnn, "Equal_xdnn_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::GreaterOrEqual, GreaterEqualXdnn,
                "GreaterEqual_xdnn_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::Greater, GreaterThanXdnn,
                "GreaterThan_xdnn_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::LessOrEqual, LessEqualXdnn,
                "LessEqual_xdnn_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::Less, LessThanXdnn,
                "LessThan_xdnn_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::FloorDiv, FloorDivXdnn,
                "FloorDiv_xdnn_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::MSELoss, MSELossXdnn,
                "MSELoss_xdnn_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::And, AndXdnn, "And_xdnn_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::Or, OrXdnn, "Or_xdnn_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::Xor, XorXdnn, "Xor_xdnn_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::Not, NotXdnn, "Not_xdnn_KUNLUN");
}; // namespace infini

#include "operators/element_wise.h"
#include "kunlun/kunlun_kernel_without_config.h"
#include "kunlun/kunlun_runtime.h"
#include "utils/operator_utils.h"

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
        checkKUNLUNError(xdnn::broadcast_add<float>(
            context->KUNLUNHandle(), (float *)aData, (float *)bData,
            (float *)cData, aDim, bDim));
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
        checkKUNLUNError(xdnn::broadcast_sub<float>(
            context->KUNLUNHandle(), (float *)aData, (float *)bData,
            (float *)cData, aDim, bDim));
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
        checkKUNLUNError(xdnn::broadcast_mul<float>(
            context->KUNLUNHandle(), (float *)aData, (float *)bData,
            (float *)cData, aDim, bDim));
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

        auto aSize = op->getInputs(0)->size();
        auto aDim = op->getInputs(0)->getDims();
        auto bSize = op->getInputs(1)->size();
        auto bDim = op->getInputs(1)->getDims();

        if (bDim.size() == 0) {
            bDim.push_back(1);
        }

        if (aSize == bSize) {
            // Do ElementWise Sub with no broadcast
            checkKUNLUNError(xdnn::div<float>(context->KUNLUNHandle(),
                                              (float *)aData, (float *)bData,
                                              (float *)cData, aSize));
        } else {
            // Do broadcast div
            Shape aligned = infer_broadcast(aDim, bDim);
            if (aligned == aDim) {
                // BData need to be broadcasted
                checkKUNLUNError(xdnn::broadcast_div<float>(
                    context->KUNLUNHandle(), (float *)aData, (float *)bData,
                    (float *)cData, aDim, bDim));
            } else {
                // Use workspace to broadcast aData
                KUNLUNPtr wks = context->getWorkspace(bSize * sizeof(float));
                checkKUNLUNError(xdnn::broadcast<float>(
                    context->KUNLUNHandle(), (float *)aData, (float *)wks, aDim,
                    bDim));
                checkKUNLUNError(xdnn::div<float>(context->KUNLUNHandle(),
                                                  (float *)wks, (float *)bData,
                                                  (float *)cData, bSize));
            }
        }
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

        checkKUNLUNError(xdnn::broadcast_pow<float>(
            context->KUNLUNHandle(), (float *)aData, (float *)bData,
            (float *)cData, aDim, bDim));
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
        checkKUNLUNError(xdnn::broadcast_max<float>(
            context->KUNLUNHandle(), (float *)aData, (float *)bData,
            (float *)cData, aDim, bDim));
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
        checkKUNLUNError(xdnn::broadcast_min<float>(
            context->KUNLUNHandle(), (float *)aData, (float *)bData,
            (float *)cData, aDim, bDim));
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
        checkKUNLUNError(xdnn::broadcast_equal<float>(
            context->KUNLUNHandle(), (float *)aData, (float *)bData,
            (bool *)wsData, aDim, bDim));
        checkKUNLUNError((xdnn::cast<bool, float>(
            context->KUNLUNHandle(), (bool *)wsData, (float *)cData, len)));
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
        checkKUNLUNError(xdnn::broadcast_greater_equal<float>(
            context->KUNLUNHandle(), (float *)aData, (float *)bData,
            (bool *)wsData, aDim, bDim));
        checkKUNLUNError((xdnn::cast<bool, float>(
            context->KUNLUNHandle(), (bool *)wsData, (float *)cData, len)));
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
        checkKUNLUNError(xdnn::broadcast_greater_than<float>(
            context->KUNLUNHandle(), (float *)aData, (float *)bData,
            (bool *)wsData, aDim, bDim));
        checkKUNLUNError((xdnn::cast<bool, float>(
            context->KUNLUNHandle(), (bool *)wsData, (float *)cData, len)));
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
        checkKUNLUNError(xdnn::broadcast_less_equal<float>(
            context->KUNLUNHandle(), (float *)aData, (float *)bData,
            (bool *)wsData, aDim, bDim));
        checkKUNLUNError((xdnn::cast<bool, float>(
            context->KUNLUNHandle(), (bool *)wsData, (float *)cData, len)));
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
        checkKUNLUNError(xdnn::broadcast_less_than<float>(
            context->KUNLUNHandle(), (float *)aData, (float *)bData,
            (bool *)wsData, aDim, bDim));
        checkKUNLUNError((xdnn::cast<bool, float>(
            context->KUNLUNHandle(), (bool *)wsData, (float *)cData, len)));
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
        checkKUNLUNError(xdnn::broadcast_floordiv<float>(
            context->KUNLUNHandle(), (float *)aData, (float *)bData,
            (float *)cData, aDim, bDim));
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
        checkKUNLUNError(xdnn::mse_loss<float>(context->KUNLUNHandle(),
                                               (float *)aData, (float *)bData,
                                               (float *)cData, len));
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
        checkKUNLUNError(xdnn::logical_and<bool>(context->KUNLUNHandle(),
                                                 (bool *)aData, (bool *)bData,
                                                 (bool *)wsData, len));
        checkKUNLUNError((xdnn::cast<bool, float>(
            context->KUNLUNHandle(), (bool *)wsData, (float *)cData, len)));
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
        checkKUNLUNError(xdnn::logical_or<bool>(context->KUNLUNHandle(),
                                                (bool *)aData, (bool *)bData,
                                                (bool *)wsData, len));
        checkKUNLUNError((xdnn::cast<bool, float>(
            context->KUNLUNHandle(), (bool *)wsData, (float *)cData, len)));
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
        checkKUNLUNError(xdnn::logical_xor<bool>(context->KUNLUNHandle(),
                                                 (bool *)aData, (bool *)bData,
                                                 (bool *)wsData, len));
        checkKUNLUNError((xdnn::cast<bool, float>(
            context->KUNLUNHandle(), (bool *)wsData, (float *)cData, len)));
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
        checkKUNLUNError(xdnn::logical_not<bool>(
            context->KUNLUNHandle(), (bool *)aData, (bool *)wsData, len));
        checkKUNLUNError((xdnn::cast<bool, float>(
            context->KUNLUNHandle(), (bool *)wsData, (float *)cData, len)));
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

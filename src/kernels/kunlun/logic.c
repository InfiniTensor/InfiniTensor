#include "operators/logic.h"
#include "kunlun/kunlun_kernel_without_config.h"
#include "kunlun/kunlun_runtime.h"
#include "utils/operator_utils.h"

namespace infini {
class AndXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<LogicOpObj>(_op); 
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        size_t len = op->getOutput()->size();
        auto dtype = op->getDType();
        KUNLUNPtr wsData = context->getWorkspace(len * dtype.getSize());

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
        auto op = as<LogicOpObj>(_op); 
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        size_t len = op->getOutput()->size();
        auto dtype = op->getDType();
        KUNLUNPtr wsData = context->getWorkspace(len * dtype.getSize());

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
        auto op = as<LogicOpObj>(_op); 
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        size_t len = op->getOutput()->size();
        auto dtype = op->getDType();
        KUNLUNPtr wsData = context->getWorkspace(len * dtype.getSize());

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
        auto op = as<LogicOpObj>(_op); 
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        size_t len = op->getOutput()->size();
        auto dtype = op->getDType();
        KUNLUNPtr wsData = context->getWorkspace(len * dtype.getSize());

        auto aDim = op->getInputs(0)->getDims();
        checkKUNLUNError(xdnn::logical_not<bool>(
            context->KUNLUNHandle(), (bool *)aData, (bool *)wsData, len));
        checkKUNLUNError((xdnn::cast<bool, float>(
            context->KUNLUNHandle(), (bool *)wsData, (float *)cData, len)));
        return;
    }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::And, AndXdnn, "And_xdnn_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::Or, OrXdnn, "Or_xdnn_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::Xor, XorXdnn, "Xor_xdnn_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::Not, NotXdnn, "Not_xdnn_KUNLUN");
}; // namespace infini

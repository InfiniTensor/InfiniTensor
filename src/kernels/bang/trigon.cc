#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"
#include "operators/unary.h"

namespace infini {
class TrigonCnnl : public BangKernelWithoutConfig {
    virtual cnnlTrigonFunctionMode_t getOpType() const = 0;
    virtual cnnlComputationPreference_t getPrefer() const = 0;
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        cnnlTensorDescriptor_t aDesc, cDesc;
        auto aDim = op->getInputs(0)->getDims();
        auto cDim = op->getOutput()->getDims();

        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            aDesc, CNNL_LAYOUT_NCHW, cnnlDataTypeConvert(op->getDType()),
            aDim.size(), aDim.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            cDesc, CNNL_LAYOUT_NCHW, cnnlDataTypeConvert(op->getDType()),
            cDim.size(), cDim.data()));

        cnnlTrigonDescriptor_t opDesc;
        checkCnnlError(cnnlCreateTrigonDescriptor(&opDesc));
        checkCnnlError(
            cnnlSetTrigonDescriptor_v2(opDesc, getOpType(), getPrefer()));

        cnnlStatus_t stat = cnnlTrigonForward(context->cnnlHandle(), opDesc,
                                              aDesc, aData, cDesc, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
        checkCnnlError(cnnlDestroyTrigonDescriptor(opDesc));
    }
};

class SinCnnl : public TrigonCnnl {
    cnnlTrigonFunctionMode_t getOpType() const override {
        return CNNL_TRIGON_SIN;
    }
    cnnlComputationPreference_t getPrefer() const override {
        return CNNL_COMPUTATION_HIGH_PRECISION;
    }
};

class CosCnnl : public TrigonCnnl {
    cnnlTrigonFunctionMode_t getOpType() const override {
        return CNNL_TRIGON_COS;
    }
    cnnlComputationPreference_t getPrefer() const override {
        return CNNL_COMPUTATION_HIGH_PRECISION;
    }
};

class TanCnnl : public TrigonCnnl {
    cnnlTrigonFunctionMode_t getOpType() const override {
        return CNNL_TRIGON_TAN;
    }
    cnnlComputationPreference_t getPrefer() const override {
        return CNNL_COMPUTATION_HIGH_PRECISION;
    }
};

class ASinCnnl : public TrigonCnnl {
    cnnlTrigonFunctionMode_t getOpType() const override {
        return CNNL_TRIGON_ASIN;
    }
    cnnlComputationPreference_t getPrefer() const override {
        return CNNL_COMPUTATION_HIGH_PRECISION;
    }
};

class ACosCnnl : public TrigonCnnl {
    cnnlTrigonFunctionMode_t getOpType() const override {
        return CNNL_TRIGON_ACOS;
    }
    cnnlComputationPreference_t getPrefer() const override {
        return CNNL_COMPUTATION_HIGH_PRECISION;
    }
};

class ATanCnnl : public TrigonCnnl {
    cnnlTrigonFunctionMode_t getOpType() const override {
        return CNNL_TRIGON_ATAN;
    }
    cnnlComputationPreference_t getPrefer() const override {
        return CNNL_COMPUTATION_HIGH_PRECISION;
    }
};

class SinHCnnl : public TrigonCnnl {
    cnnlTrigonFunctionMode_t getOpType() const override {
        return CNNL_TRIGON_SINH;
    }
    cnnlComputationPreference_t getPrefer() const override {
        return CNNL_COMPUTATION_HIGH_PRECISION;
    }
};

class CosHCnnl : public TrigonCnnl {
    cnnlTrigonFunctionMode_t getOpType() const override {
        return CNNL_TRIGON_COSH;
    }
    cnnlComputationPreference_t getPrefer() const override {
        return CNNL_COMPUTATION_HIGH_PRECISION;
    }
};

class TanHCnnl : public TrigonCnnl {
    cnnlTrigonFunctionMode_t getOpType() const override {
        return CNNL_TRIGON_TANH;
    }
    cnnlComputationPreference_t getPrefer() const override {
        return CNNL_COMPUTATION_HIGH_PRECISION;
    }
};

class ASinHCnnl : public TrigonCnnl {
    cnnlTrigonFunctionMode_t getOpType() const override {
        return CNNL_TRIGON_ASINH;
    }
    cnnlComputationPreference_t getPrefer() const override {
        return CNNL_COMPUTATION_HIGH_PRECISION;
    }
};

class ACosHCnnl : public TrigonCnnl {
    cnnlTrigonFunctionMode_t getOpType() const override {
        return CNNL_TRIGON_ACOSH;
    }
    cnnlComputationPreference_t getPrefer() const override {
        return CNNL_COMPUTATION_HIGH_PRECISION;
    }
};

class ATanHCnnl : public TrigonCnnl {
    cnnlTrigonFunctionMode_t getOpType() const override {
        return CNNL_TRIGON_ATANH;
    }
    cnnlComputationPreference_t getPrefer() const override {
        return CNNL_COMPUTATION_HIGH_PRECISION;
    }
};

REGISTER_KERNEL(Device::BANG, OpType::Sin, SinCnnl, "Sin_cnnl_BANG");
REGISTER_KERNEL(Device::BANG, OpType::Cos, CosCnnl, "Cos_cnnl_BANG");
REGISTER_KERNEL(Device::BANG, OpType::Tan, TanCnnl, "Tan_cnnl_BANG");
REGISTER_KERNEL(Device::BANG, OpType::Asin, ASinCnnl, "ASin_cnnl_BANG");
REGISTER_KERNEL(Device::BANG, OpType::Acos, ACosCnnl, "ACos_cnnl_BANG");
REGISTER_KERNEL(Device::BANG, OpType::Atan, ATanCnnl, "ATan_cnnl_BANG");
REGISTER_KERNEL(Device::BANG, OpType::Sinh, SinHCnnl, "SinH_cnnl_BANG");
REGISTER_KERNEL(Device::BANG, OpType::Cosh, CosHCnnl, "CosH_cnnl_BANG");
REGISTER_KERNEL(Device::BANG, OpType::Tanh, TanHCnnl, "TanH_cnnl_BANG");
REGISTER_KERNEL(Device::BANG, OpType::Asinh, ASinHCnnl, "ASinH_cnnl_BANG");
REGISTER_KERNEL(Device::BANG, OpType::Acosh, ACosHCnnl, "ACosH_cnnl_BANG");
REGISTER_KERNEL(Device::BANG, OpType::Atanh, ATanHCnnl, "ATanH_cnnl_BANG");

}; // namespace infini

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
        auto dim = op->getInputs(0)->getDims();
        if (dim.size() != 4)
            IT_TODO_HALT();

        int dim_array[4] = {dim[0], dim[1], dim[2], dim[3]};
        // get inputs
        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));

        // get outputs
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));

        // get op descriptor
        cnnlTrigonDescriptor_t opDesc;
        checkCnnlError(cnnlCreateTrigonDescriptor(&opDesc));
        checkCnnlError(cnnlSetTrigonDescriptor(opDesc, getOpType()));

        cnnlStatus_t stat = cnnlTrigonForward(context->cnnlHandle(), opDesc,
                                              aDesc, aData, cDesc, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        // Destories in BANG does not require sync. But cnnl does not state
        // whether sync is required before destories.
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

REGISTER_KERNEL(Device::BANG, OpType::Sin, DataType::Float32, SinCnnl,
                "Sin_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::Cos, DataType::Float32, CosCnnl,
                "Cos_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::Tan, DataType::Float32, TanCnnl,
                "Tan_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::ASin, DataType::Float32, ASinCnnl,
                "ASin_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::ACos, DataType::Float32, ACosCnnl,
                "ACos_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::ATan, DataType::Float32, ATanCnnl,
                "ATan_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::SinH, DataType::Float32, SinHCnnl,
                "SinH_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::CosH, DataType::Float32, CosHCnnl,
                "CosH_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::TanH, DataType::Float32, TanHCnnl,
                "TanH_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::ASinH, DataType::Float32, ASinHCnnl,
                "ASinH_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::ACosH, DataType::Float32, ACosHCnnl,
                "ACosH_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::ATanH, DataType::Float32, ATanHCnnl,
                "ATanH_cnnl_BANG_Float32");

}; // namespace infini

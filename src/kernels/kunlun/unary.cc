#include "operators/unary.h"
#include "kunlun/kunlun_kernel_without_config.h"
#include "kunlun/kunlun_runtime.h"

namespace infini {
class ReluXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();

        auto ret = xdnn::relu<float>(context->KUNLUNHandle(), (float *)aData,
                                     (float *)cData, len);
        assert(ret == 0);
        return;
    }
};

class LeakyReluXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<LeakyReluObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();
        auto alpha = op->getAlpha();

        auto ret = xdnn::leaky_relu<float>(context->KUNLUNHandle(),
                                           (float *const)aData, (float *)cData,
                                           len, alpha);
        assert(ret == 0);
        return;
    }
};

class SigmoidXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();

        auto ret = xdnn::sigmoid<float>(context->KUNLUNHandle(), (float *)aData,
                                        (float *)cData, len);
        assert(ret == 0);
        return;
    }
};

class TanhXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();

        auto ret = xdnn::tanh<float>(context->KUNLUNHandle(), (float *)aData,
                                     (float *)cData, len);
        assert(ret == 0);
        return;
    }
};

class HardSwishXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();

        auto ret = xdnn::hard_swish<float>(context->KUNLUNHandle(),
                                           (float *)aData, (float *)cData, len);
        assert(ret == 0);
        return;
    }
};

class HardSigmoidXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();

        // Slop set to 0.2 as default
        auto ret = xdnn::hard_sigmoid<float>(
            context->KUNLUNHandle(), (float *)aData, (float *)cData, len, 0.2);
        assert(ret == 0);
        return;
    }
};

class SquareXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();

        auto ret = xdnn::square<float>(context->KUNLUNHandle(), (float *)aData,
                                       (float *)cData, len);
        assert(ret == 0);
        return;
    }
};

class SqrtXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();

        auto ret = xdnn::sqrt<float>(context->KUNLUNHandle(), (float *)aData,
                                     (float *)cData, len);
        assert(ret == 0);
        return;
    }
};

class RsqrtXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();

        auto ret = xdnn::rsqrt<float>(context->KUNLUNHandle(), (float *)aData,
                                      (float *)cData, len);
        assert(ret == 0);
        return;
    }
};

class ExpXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();

        auto ret = xdnn::exp<float>(context->KUNLUNHandle(), (float *)aData,
                                    (float *)cData, len);
        assert(ret == 0);
        return;
    }
};

class CeilXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();

        auto ret = xdnn::ceil<float>(context->KUNLUNHandle(), (float *)aData,
                                     (float *)cData, len);
        assert(ret == 0);
        return;
    }
};

class ClipXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ClipObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();
        float min = op->getMin().value();
        float max = op->getMax().value();

        auto ret = xdnn::clip<float>(context->KUNLUNHandle(), (float *)aData,
                                     (float *)cData, len, min, max);
        assert(ret == 0);
        return;
    }
};

class FloorXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();

        auto ret = xdnn::floor<float>(context->KUNLUNHandle(), (float *)aData,
                                      (float *)cData, len);
        assert(ret == 0);
        return;
    }
};

class NegXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();

        auto ret = xdnn::neg<float>(context->KUNLUNHandle(), (float *)aData,
                                    (float *)cData, len);
        assert(ret == 0);
        return;
    }
};

class CopyXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &op,
                 const RuntimeObj *_context) const override {
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();

        auto ret = xdnn::copy<float>(context->KUNLUNHandle(), (float *)aData,
                                     (float *)cData, len);
        assert(ret == 0);
        return;
    }
};

class ReciprocalXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();

        auto ret = xdnn::reciprocal<float>(context->KUNLUNHandle(),
                                           (float *)aData, (float *)cData, len);
        assert(ret == 0);
        return;
    }
};

class AbsXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();

        auto ret = xdnn::abs<float>(context->KUNLUNHandle(), (float *)aData,
                                    (float *)cData, len);
        assert(ret == 0);
        return;
    }
};

class ATanXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();

        auto ret = xdnn::arctan<float>(context->KUNLUNHandle(), (float *)aData,
                                       (float *)cData, len);
        assert(ret == 0);
        return;
    }
};

class LogXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<LogObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);
        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto aDim = op->getInputs(0)->getDims();
        std::vector<int> divDim = {
            1,
        };
        auto len = op->getInputs(0)->size();
        auto dtype = op->getDType();
        // get ptr of tempspace
        KUNLUNPtr temp = context->getWorkspace(len * dtype.getSize());
        LogObj::LogType type = op->getType();
        // get output of xpu::api::loge(x)
        auto ret = xdnn::log<float>(context->KUNLUNHandle(), (float *)aData,
                                    (float *)temp, len);
        // get ptr of divider
        KUNLUNPtr dd = context->getWorkspace(1 * dtype.getSize());
        // choose from logE, log2, log10
        switch (type) {
            float constant;
        case LogObj::LogE:
            // if use loge, copy from temp to cData
            ret = xdnn::copy<float>(context->KUNLUNHandle(), (float *)temp,
                                    (float *)cData, len);
            break;
        case LogObj::Log2:
            constant = std::log(2);
            context->copyBlobFromCPU(dd, &constant, sizeof(float));
            ret = xdnn::broadcast_div<float>(context->KUNLUNHandle(),
                                             (float *)temp, (float *)dd,
                                             (float *)cData, aDim, divDim);
            break;
        case LogObj::Log10:
            constant = std::log(10);
            context->copyBlobFromCPU(dd, &constant, sizeof(float));
            ret = xdnn::broadcast_div<float>(context->KUNLUNHandle(),
                                             (float *)temp, (float *)dd,
                                             (float *)cData, aDim, divDim);
            break;
        default:
            printf("LogType not support!");
            break;
        }
        assert(ret == 0);
        return;
    }
};

class CosXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<CosObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);
        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();
        auto ret = xdnn::cos<float>(context->KUNLUNHandle(), (float *)aData,
                                    (float *)cData, len);

        assert(ret == 0);
        return;
    }
};

class SinXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<SinObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);
        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();
        auto ret = xdnn::sin<float>(context->KUNLUNHandle(), (float *)aData,
                                    (float *)cData, len);

        assert(ret == 0);
        return;
    }
};

class TanXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<TanObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);
        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();
        auto ret = xdnn::tan<float>(context->KUNLUNHandle(), (float *)aData,
                                    (float *)cData, len);

        assert(ret == 0);
        return;
    }
};

class SinhXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<SinHObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);
        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();
        auto ret = xdnn::sinh<float>(context->KUNLUNHandle(), (float *)aData,
                                     (float *)cData, len);

        assert(ret == 0);
        return;
    }
};

class CoshXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<CosHObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);
        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();
        auto ret = xdnn::cosh<float>(context->KUNLUNHandle(), (float *)aData,
                                     (float *)cData, len);

        assert(ret == 0);
        return;
    }
};

class ErfXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ErfObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);
        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();
        auto ret = xdnn::erf<float>(context->KUNLUNHandle(), (float *)aData,
                                    (float *)cData, len);

        assert(ret == 0);
        return;
    }
};

class ACosXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ACosObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);
        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();
        auto ret = xdnn::arccos<float>(context->KUNLUNHandle(), (float *)aData,
                                       (float *)cData, len);

        assert(ret == 0);
        return;
    }
};

class ACoshXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ACosHObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);
        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();
        auto ret = xdnn::acosh<float>(context->KUNLUNHandle(), (float *)aData,
                                      (float *)cData, len);

        assert(ret == 0);
        return;
    }
};

class ASinXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ASinObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);
        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();
        auto ret = xdnn::arcsin<float>(context->KUNLUNHandle(), (float *)aData,
                                       (float *)cData, len);

        assert(ret == 0);
        return;
    }
};

class ASinhXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ASinHObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);
        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();
        auto ret = xdnn::asinh<float>(context->KUNLUNHandle(), (float *)aData,
                                      (float *)cData, len);

        assert(ret == 0);
        return;
    }
};

class ATanhXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ATanHObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);
        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();
        auto ret = xdnn::atanh<float>(context->KUNLUNHandle(), (float *)aData,
                                      (float *)cData, len);

        assert(ret == 0);
        return;
    }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::Relu, ReluXdnn, "Relu_xdnn_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::LeakyRelu, LeakyReluXdnn,
                "LeakyRelu_xdnn_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::Sigmoid, SigmoidXdnn,
                "Sigmoid_xdnn_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::Tanh, TanhXdnn, "Tanh_xdnn_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::Square, SquareXdnn,
                "Square_xdnn_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::Sqrt, SqrtXdnn, "Sqrt_xdnn_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::Rsqrt, RsqrtXdnn, "Rsqrt_xdnn_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::Exp, ExpXdnn, "Exp_xdnn_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::Ceil, CeilXdnn, "Ceil_xdnn_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::Clip, ClipXdnn, "Clip_xdnn_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::Floor, FloorXdnn, "Floor_xdnn_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::Neg, NegXdnn, "Neg_xdnn_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::Reciprocal, ReciprocalXdnn,
                "Reciprocal_xdnn_KUNLUN");

REGISTER_KERNEL(Device::KUNLUN, OpType::Reshape, CopyXdnn, "Reshape_xdnn");
REGISTER_KERNEL(Device::KUNLUN, OpType::Flatten, CopyXdnn, "Flatten_xdnn");
REGISTER_KERNEL(Device::KUNLUN, OpType::Identity, CopyXdnn, "Identity_xdnn");
REGISTER_KERNEL(Device::KUNLUN, OpType::Squeeze, CopyXdnn, "Squeeze_xdnn");
REGISTER_KERNEL(Device::KUNLUN, OpType::Abs, AbsXdnn, "Abs_xdnn");
REGISTER_KERNEL(Device::KUNLUN, OpType::Atan, ATanXdnn, "Atan_xdnn");
REGISTER_KERNEL(Device::KUNLUN, OpType::Log, LogXdnn, "Log_xdnn");
REGISTER_KERNEL(Device::KUNLUN, OpType::Cos, CosXdnn, "Cos_xdnn");
REGISTER_KERNEL(Device::KUNLUN, OpType::Sin, SinXdnn, "Sin_xdnn");
REGISTER_KERNEL(Device::KUNLUN, OpType::Tan, TanXdnn, "Tan_xdnn");
REGISTER_KERNEL(Device::KUNLUN, OpType::Sinh, SinhXdnn, "Sinh_xdnn");
REGISTER_KERNEL(Device::KUNLUN, OpType::Cosh, CoshXdnn, "Cosh_xdnn");
REGISTER_KERNEL(Device::KUNLUN, OpType::Erf, ErfXdnn, "Erf_xdnn");
REGISTER_KERNEL(Device::KUNLUN, OpType::Acos, ACosXdnn, "ACos_xdnn");
REGISTER_KERNEL(Device::KUNLUN, OpType::Acosh, ACoshXdnn, "ACosh_xdnn");
REGISTER_KERNEL(Device::KUNLUN, OpType::Asin, ASinXdnn, "ASin_xdnn");
REGISTER_KERNEL(Device::KUNLUN, OpType::Asinh, ASinhXdnn, "ASinh_xdnn");
REGISTER_KERNEL(Device::KUNLUN, OpType::Atanh, ATanhXdnn, "ATanh_xdnn");
REGISTER_KERNEL(Device::KUNLUN, OpType::HardSwish, HardSwishXdnn,
                "HardSwish_xdnn");
REGISTER_KERNEL(Device::KUNLUN, OpType::HardSigmoid, HardSigmoidXdnn,
                "HardSigmoid_xdnn");
}; // namespace infini

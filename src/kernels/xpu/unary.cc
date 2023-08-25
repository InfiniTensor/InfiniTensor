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

        auto ret = baidu::xpu::api::relu<float>(
            context->XPUHandle(), (float *)aData, (float *)cData, len);
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

        auto ret = baidu::xpu::api::sigmoid<float>(
            context->XPUHandle(), (float *)aData, (float *)cData, len);
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

        auto ret = baidu::xpu::api::tanh<float>(
            context->XPUHandle(), (float *)aData, (float *)cData, len);
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

        auto ret = baidu::xpu::api::square<float>(
            context->XPUHandle(), (float *)aData, (float *)cData, len);
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

        auto ret = baidu::xpu::api::sqrt<float>(
            context->XPUHandle(), (float *)aData, (float *)cData, len);
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

        auto ret = baidu::xpu::api::rsqrt<float>(
            context->XPUHandle(), (float *)aData, (float *)cData, len);
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

        auto ret = baidu::xpu::api::exp<float>(
            context->XPUHandle(), (float *)aData, (float *)cData, len);
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

        auto ret = baidu::xpu::api::ceil<float>(
            context->XPUHandle(), (float *)aData, (float *)cData, len);
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

        auto ret =
            baidu::xpu::api::clip<float>(context->XPUHandle(), (float *)aData,
                                         (float *)cData, len, min, max);
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

        auto ret = baidu::xpu::api::floor<float>(
            context->XPUHandle(), (float *)aData, (float *)cData, len);
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

        auto ret = baidu::xpu::api::neg<float>(
            context->XPUHandle(), (float *)aData, (float *)cData, len);
        assert(ret == 0);
        return;
    }
};

class CopyXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &op,
                 const RuntimeObj *_context) const override {
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();

        auto ret = baidu::xpu::api::copy<float>(
            context->XPUHandle(), (float *)aData, (float *)cData, len);
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

        auto ret = baidu::xpu::api::reciprocal<float>(
            context->XPUHandle(), (float *)aData, (float *)cData, len);
        assert(ret == 0);
        return;
    }
};

class AbsXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();

        auto ret = baidu::xpu::api::abs<float>(
            context->XPUHandle(), (float *)aData, (float *)cData, len);
        assert(ret == 0);
        return;
    }
};

class ATanXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();

        auto ret = baidu::xpu::api::arctan<float>(
            context->XPUHandle(), (float *)aData, (float *)cData, len);
        assert(ret == 0);
        return;
    }
};

class LogXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<LogObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);
        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto aDim = op->getInputs(0)->getDims();
        std::vector<int> divDim = {
            1,
        };
        auto len = op->getInputs(0)->size();
        // get ptr of tempspace
        XPUPtr temp = context->getWorkspace(len * sizeof(float));
        LogObj::LogType type = op->getType();
        // get output of xpu::api::loge(x)
        auto ret = baidu::xpu::api::log<float>(
            context->XPUHandle(), (float *)aData, (float *)temp, len);
        // get ptr of divider
        XPUPtr dd =
            (float *)(context->getWorkspace((1 + len) * sizeof(float))) +
            len * sizeof(float);
        // printf("=================> ret after xpu::api::log<float>: %d\n",
        // ret); choose from logE, log2, log10
        switch (type) {
            float constant;
        case LogObj::LogE:
            // if use loge, copy from temp to cData
            ret = baidu::xpu::api::copy<float>(
                context->XPUHandle(), (float *)temp, (float *)cData, len);
            break;
        case LogObj::Log2:
            constant = std::log(2);
            context->copyBlobFromCPU(dd, &constant, sizeof(float));
            ret = baidu::xpu::api::broadcast_div<float>(
                context->XPUHandle(), (float *)temp, (float *)dd,
                (float *)cData, aDim, divDim);
            break;
        case LogObj::Log10:
            constant = std::log(10);
            context->copyBlobFromCPU(dd, &constant, sizeof(float));
            ret = baidu::xpu::api::broadcast_div<float>(
                context->XPUHandle(), (float *)temp, (float *)dd,
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

class CosXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<CosObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);
        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();
        auto ret = baidu::xpu::api::cos<float>(
            context->XPUHandle(), (float *)aData, (float *)cData, len);

        assert(ret == 0);
        return;
    }
};

class SinXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<SinObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);
        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();
        auto ret = baidu::xpu::api::sin<float>(
            context->XPUHandle(), (float *)aData, (float *)cData, len);

        assert(ret == 0);
        return;
    }
};

class TanXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<TanObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);
        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();
        auto ret = baidu::xpu::api::tan<float>(
            context->XPUHandle(), (float *)aData, (float *)cData, len);

        assert(ret == 0);
        return;
    }
};

class SinhXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<SinHObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);
        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();
        auto ret = baidu::xpu::api::sinh<float>(
            context->XPUHandle(), (float *)aData, (float *)cData, len);

        assert(ret == 0);
        return;
    }
};

class CoshXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<CosHObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);
        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();
        auto ret = baidu::xpu::api::cosh<float>(
            context->XPUHandle(), (float *)aData, (float *)cData, len);

        assert(ret == 0);
        return;
    }
};

class ErfXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ErfObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);
        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();
        auto ret = baidu::xpu::api::erf<float>(
            context->XPUHandle(), (float *)aData, (float *)cData, len);

        assert(ret == 0);
        return;
    }
};

class ACosXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ACosObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);
        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();
        auto ret = baidu::xpu::api::arccos<float>(
            context->XPUHandle(), (float *)aData, (float *)cData, len);

        assert(ret == 0);
        return;
    }
};

class ACoshXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ACosHObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);
        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();
        auto ret = baidu::xpu::api::acosh<float>(
            context->XPUHandle(), (float *)aData, (float *)cData, len);

        assert(ret == 0);
        return;
    }
};

class ASinXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ASinObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);
        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();
        auto ret = baidu::xpu::api::arcsin<float>(
            context->XPUHandle(), (float *)aData, (float *)cData, len);

        assert(ret == 0);
        return;
    }
};

class ASinhXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ASinHObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);
        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();
        auto ret = baidu::xpu::api::asinh<float>(
            context->XPUHandle(), (float *)aData, (float *)cData, len);

        assert(ret == 0);
        return;
    }
};

class ATanhXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ATanHObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);
        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();
        auto ret = baidu::xpu::api::atanh<float>(
            context->XPUHandle(), (float *)aData, (float *)cData, len);

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
REGISTER_KERNEL(Device::XPU, OpType::Reciprocal, DataType::Float32,
                ReciprocalXdnn, "Reciprocal_xdnn_XPU_Float32");

REGISTER_KERNEL(Device::XPU, OpType::Reshape, DataType::Float32, CopyXdnn,
                "Reshape_xdnn_Float32");
REGISTER_KERNEL(Device::XPU, OpType::Flatten, DataType::Float32, CopyXdnn,
                "Flatten_xdnn_Float32");
REGISTER_KERNEL(Device::XPU, OpType::Identity, DataType::Float32, CopyXdnn,
                "Identity_xdnn_Float32");
REGISTER_KERNEL(Device::XPU, OpType::Abs, DataType::Float32, AbsXdnn,
                "Abs_xdnn_Float32");
REGISTER_KERNEL(Device::XPU, OpType::Atan, DataType::Float32, ATanXdnn,
                "Atan_xdnn_Float32");
REGISTER_KERNEL(Device::XPU, OpType::Log, DataType::Float32, LogXdnn,
                "Log_xdnn_Float32");
REGISTER_KERNEL(Device::XPU, OpType::Cos, DataType::Float32, CosXdnn,
                "Cos_xdnn_Float32");
REGISTER_KERNEL(Device::XPU, OpType::Sin, DataType::Float32, SinXdnn,
                "Sin_xdnn_Float32");
REGISTER_KERNEL(Device::XPU, OpType::Tan, DataType::Float32, TanXdnn,
                "Tan_xdnn_Float32");
REGISTER_KERNEL(Device::XPU, OpType::Sinh, DataType::Float32, SinhXdnn,
                "Sinh_xdnn_Float32");
REGISTER_KERNEL(Device::XPU, OpType::Cosh, DataType::Float32, CoshXdnn,
                "Cosh_xdnn_Float32");
REGISTER_KERNEL(Device::XPU, OpType::Erf, DataType::Float32, ErfXdnn,
                "Erf_xdnn_Float32");
REGISTER_KERNEL(Device::XPU, OpType::Acos, DataType::Float32, ACosXdnn,
                "ACos_xdnn_Float32");
REGISTER_KERNEL(Device::XPU, OpType::Acosh, DataType::Float32, ACoshXdnn,
                "ACosh_xdnn_Float32");
REGISTER_KERNEL(Device::XPU, OpType::Asin, DataType::Float32, ASinXdnn,
                "ASin_xdnn_Float32");
REGISTER_KERNEL(Device::XPU, OpType::Asinh, DataType::Float32, ASinhXdnn,
                "ASinh_xdnn_Float3 2");
REGISTER_KERNEL(Device::XPU, OpType::Atanh, DataType::Float32, ATanhXdnn,
                "ATanh_xdnn_Float32");
}; // namespace infini

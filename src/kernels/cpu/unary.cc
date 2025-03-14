#include "operators/unary.h"
#include "core/constants.h"
#include "core/kernel.h"
#include "operators/softmax.h"

namespace infini {
class NativeUnary : public CpuKernelWithoutConfig {
    template <typename T> static T reluCompute(T val) {
        return std::max(T(0), val);
    }

    template <typename T> static T sigmoidCompute(T val) {
        return 1 / (1 + pow(E_CONSTANT, -val));
    }

    template <typename T> static T hardSigmoidCompute(T val) {
        return std::max(T(0), std::min(T(1), T(0.2) * val + T(0.5)));
    }

    template <typename T> static T hardSwishCompute(T val) {
        return val *
               std::max(T(0), std::min(T(1), val * T(1.0 / 6.0) + T(0.5)));
    }

    template <typename T> static T tanhCompute(T val) {
        return (pow(E_CONSTANT, val) - pow(E_CONSTANT, -val)) /
               (pow(E_CONSTANT, val) + pow(E_CONSTANT, -val));
    }

    template <typename T> static T absCompute(T val) {
        return val < 0 ? -val : val;
    }

    template <typename T> static T sqrtCompute(T val) { return std::sqrt(val); }

    template <typename T> static T cosCompute(T val) { return std::cos(val); }

    template <typename T> static T sinCompute(T val) { return std::sin(val); }

    template <typename T> static T tanCompute(T val) { return std::tan(val); }

    template <typename T> static T sinhCompute(T val) { return std::sinh(val); }

    template <typename T> static T coshCompute(T val) { return std::cosh(val); }

    template <typename T> static T geluCompute(T val) {
        return 0.5 * val * (1 + std::erf(val / std::sqrt(2)));
    }

    template <typename T> static T siluCompute(T val) {
        return val / (1 + pow(E_CONSTANT, -val));
    }

    template <typename T> static T erfCompute(T val) { return std::erf(val); }

    template <typename T> static T aCosCompute(T val) { return std::acos(val); }

    template <typename T> static T aCoshCompute(T val) {
        return std::acosh(val);
    }

    template <typename T> static T aSinCompute(T val) { return std::asin(val); }

    template <typename T> static T aSinhCompute(T val) {
        return std::asinh(val);
    }
    template <typename T> static T aTanCompute(T val) { return std::atan(val); }

    template <typename T> static T aTanhCompute(T val) {
        return std::atanh(val);
    }
    template <typename T> static T negCompute(T val) { return -val; }

    template <typename T>
    void doCompute(const Operator &_op, const RuntimeObj *context) const {
        auto op = as<UnaryObj>(_op);
        T *inptr = op->getInputs(0)->getRawDataPtr<T *>();
        T *outptr = op->getOutput()->getRawDataPtr<T *>();

        auto outDim = op->getOutput()->getDims();
        auto n = op->getOutput()->size();

        T (*_doCompute)(T val);
        switch (op->getOpType().underlying()) {
        case OpType::Relu:
            _doCompute = reluCompute<T>;
            break;
        case OpType::Gelu:
            _doCompute = geluCompute<T>;
            break;
        case OpType::Silu:
            _doCompute = siluCompute<T>;
            break;
        case OpType::Sigmoid:
            _doCompute = sigmoidCompute<T>;
            break;
        case OpType::HardSigmoid:
            _doCompute = hardSigmoidCompute<T>;
            break;
        case OpType::HardSwish:
            _doCompute = hardSwishCompute<T>;
            break;
        case OpType::Tanh:
            _doCompute = tanhCompute<T>;
            break;
        case OpType::Abs:
            _doCompute = absCompute<T>;
            break;
        case OpType::Sqrt:
            _doCompute = sqrtCompute<T>;
            break;
        case OpType::Erf:
            _doCompute = erfCompute<T>;
            break;
        case OpType::Neg:
            _doCompute = negCompute<T>;
            break;
        case OpType::Cos:
            _doCompute = cosCompute<T>;
            break;
        case OpType::Sin:
            _doCompute = sinCompute<T>;
            break;
        case OpType::Tan:
            _doCompute = tanCompute<T>;
            break;
        case OpType::Sinh:
            _doCompute = sinhCompute<T>;
            break;
        case OpType::Cosh:
            _doCompute = coshCompute<T>;
            break;
        case OpType::Acos:
            _doCompute = aCosCompute<T>;
            break;
        case OpType::Asin:
            _doCompute = aSinCompute<T>;
            break;
        case OpType::Asinh:
            _doCompute = aSinhCompute<T>;
            break;
        case OpType::Atan:
            _doCompute = aTanCompute<T>;
            break;
        case OpType::Atanh:
            _doCompute = aTanhCompute<T>;
            break;
        case OpType::Acosh:
            _doCompute = aCoshCompute<T>;
            break;
        default:
            IT_TODO_HALT();
        }

        for (size_t offset = 0; offset < n; offset++) {
            outptr[offset] = _doCompute(inptr[offset]);
        }
    }

    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
#define CASE(N)                                                                \
    case N:                                                                    \
        doCompute<DT<N>::t>(_op, context)

        int dataTypeIdx = _op->getDType().getIndex();
        switch (dataTypeIdx) {
            CASE(1); // DataType::Float32
            break;
            CASE(12); // DataType::UInt32
            break;
        default:
            IT_TODO_HALT();
        }
    }
};

class NaiveSoftmax : public CpuKernelWithoutConfig {
    template <typename T>
    void doCompute(const Operator &_op, const RuntimeObj *context) const {
        auto op = as<SoftmaxObj>(_op);
        T *inptr = op->getInputs(0)->getRawDataPtr<T *>();
        T *outptr = op->getOutput()->getRawDataPtr<T *>();

        auto outDim = op->getOutput()->getDims();
        auto n = op->getOutput()->size();
        auto sum = T(0);
        for (size_t offset = 0; offset < n; offset++) {
            sum += pow(E_CONSTANT, inptr[offset]);
        }
        for (size_t offset = 0; offset < n; offset++) {
            outptr[offset] = pow(E_CONSTANT, inptr[offset]) / sum;
        }
    }

    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
#define CASE(N)                                                                \
    case N:                                                                    \
        doCompute<DT<N>::t>(_op, context)

        int dataTypeIdx = _op->getDType().getIndex();
        switch (dataTypeIdx) {
            CASE(1); // DataType::Float32
            break;
            CASE(12); // DataType::UInt32
            break;
        default:
            IT_TODO_HALT();
        }
    }
};

class Clip : public CpuKernelWithoutConfig {
    template <typename T>
    void doCompute(const Operator &_op, const RuntimeObj *context) const {
        auto op = as<ClipObj>(_op);
        T *inptr = op->getInputs(0)->getRawDataPtr<T *>();
        T *outptr = op->getOutput()->getRawDataPtr<T *>();
        auto minValue = op->getMin();
        auto maxValue = op->getMax();

        auto n = op->getOutput()->size();
        for (size_t offset = 0; offset < n; offset++) {
            auto val = *inptr++;
            *outptr++ = (minValue && val < *minValue)   ? *minValue
                        : (maxValue && val > *maxValue) ? *maxValue
                                                        : val;
        }
    }

    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
#define CASE(N)                                                                \
    case N:                                                                    \
        doCompute<DT<N>::t>(_op, context)

        int dataTypeIdx = _op->getDType().getIndex();
        switch (dataTypeIdx) {
            CASE(1); // DataType::Float32
            break;
            CASE(12); // DataType::UInt32
            break;
        default:
            IT_TODO_HALT();
        }
    }
};

class Log : public CpuKernelWithoutConfig {
    template <typename T>
    void doCompute(const Operator &_op, const RuntimeObj *context) const {
        auto op = as<LogObj>(_op);
        T *inptr = op->getInputs(0)->getRawDataPtr<T *>();
        T *outptr = op->getOutput()->getRawDataPtr<T *>();
        auto logType = op->getType(); // get log type

        auto len = op->getOutput()->size();
        for (size_t offset = 0; offset < len; offset++) {
            T res;
            auto val = *inptr++;
            switch (logType) {
            case LogObj::LogE:
                res = std::log(val);
                *outptr++ = res;
                break;
            case LogObj::Log2:
                res = std::log2(val);
                *outptr++ = res;
                break;
            case LogObj::Log10:
                res = std::log10(val);
                *outptr++ = res;
                break;
            default:
                printf("LogType not Defined");
                break;
            }
        }
    }

    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
#define CASE(N)                                                                \
    case N:                                                                    \
        doCompute<DT<N>::t>(_op, context)

        int dataTypeIdx = _op->getDType().getIndex();
        switch (dataTypeIdx) {
            CASE(1); // DataType::Float32
            break;
            CASE(12); // DataType::UInt32
            break;
        default:
            IT_TODO_HALT();
        }
    }
};

// REGISTER_KERNEL(Device::CPU, OpType::Relu, NativeUnary, "reluNaive_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Gelu, NativeUnary, "geluNaive_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Silu, NativeUnary, "siluNaive_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Sigmoid, NativeUnary, "sigmoidNaive_CPU");
REGISTER_KERNEL(Device::CPU, OpType::HardSigmoid, NativeUnary,
                "hardSigmoidNaive_CPU");
REGISTER_KERNEL(Device::CPU, OpType::HardSwish, NativeUnary,
                "hardSwishNaive_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Tanh, NativeUnary, "tanhNaive_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Abs, NativeUnary, "absNaive_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Sqrt, NativeUnary, "sqrtNaive_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Erf, NativeUnary, "erfNaive_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Neg, NativeUnary, "negNaive_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Cos, NativeUnary, "Cos_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Sin, NativeUnary, "Sin_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Tan, NativeUnary, "Tan_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Sinh, NativeUnary, "Sinh_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Cosh, NativeUnary, "Cosh_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Acos, NativeUnary, "ACos_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Acosh, NativeUnary, "ACosh_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Asin, NativeUnary, "ASin_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Asinh, NativeUnary, "ASinh_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Atan, NativeUnary, "Atan_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Atanh, NativeUnary, "ATanh_CPU");

REGISTER_KERNEL(Device::CPU, OpType::Softmax, NaiveSoftmax, "softmaxNaive_CPU");
//REGISTER_KERNEL(Device::CPU, OpType::Clip, Clip, "Clip_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Log, Log, "Log_CPU");
}; // namespace infini

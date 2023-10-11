#include "operators/unary.h"
#include "core/constants.h"
#include "core/kernel.h"

namespace infini {
template <typename T> class NativeUnary : public CpuKernelWithoutConfig {
    virtual T doCompute(T val) const = 0;
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<UnaryObj>(_op);
        T *inptr = op->getInputs(0)->getRawDataPtr<T *>();
        T *outptr = op->getOutput()->getRawDataPtr<T *>();

        auto outDim = op->getOutput()->getDims();
        auto n = op->getOutput()->size();
        for (size_t offset = 0; offset < n; offset++) {
            outptr[offset] = doCompute(inptr[offset]);
        }
    }
};

template <typename T> class NaiveSoftmax : public CpuKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<UnaryObj>(_op);
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
};

template <typename T> class NaiveRelu : public NativeUnary<T> {
    T doCompute(T val) const override { return std::max(T(0), val); }
};
template <typename T> class NaiveSigmoid : public NativeUnary<T> {
    T doCompute(T val) const override {
        return 1 / (1 + pow(E_CONSTANT, -val));
    }
};
template <typename T> class NaiveHardSigmoid : public NativeUnary<T> {
    T doCompute(T val) const override {
        return std::max(T(0), std::min(T(1), T(0.2) * val + T(0.5)));
    }
};
template <typename T> class NaiveHardSwish : public NativeUnary<T> {
    T doCompute(T val) const override {
        return val *
               std::max(T(0), std::min(T(1), val * T(1.0 / 6.0) + T(0.5)));
    }
};
template <typename T> class NaiveTanh : public NativeUnary<T> {
    T doCompute(T val) const override {
        return (pow(E_CONSTANT, val) - pow(E_CONSTANT, -val)) /
               (pow(E_CONSTANT, val) + pow(E_CONSTANT, -val));
    }
};
template <typename T> class NaiveAbs : public NativeUnary<T> {
    T doCompute(T val) const override { return val < 0 ? -val : val; }
};

template <typename T> class NaiveSqrt : public NativeUnary<T> {
    T doCompute(T val) const override { return std::sqrt(val); }
};

template <typename T> class NaiveCos : public NativeUnary<T> {
    T doCompute(T val) const override { return std::cos(val); }
};

template <typename T> class NaiveSin : public NativeUnary<T> {
    T doCompute(T val) const override { return std::sin(val); }
};

template <typename T> class NaiveTan : public NativeUnary<T> {
    T doCompute(T val) const override { return std::tan(val); }
};

template <typename T> class NaiveSinh : public NativeUnary<T> {
    T doCompute(T val) const override { return std::sinh(val); }
};

template <typename T> class NaiveCosh : public NativeUnary<T> {
    T doCompute(T val) const override { return std::cosh(val); }

    template <typename T> class NaiveGelu : public NativeUnary<T> {
        T doCompute(T val) const override {
            return 0.5 * val * (1 + std::erf(val / std::sqrt(2)));
        }
    };

    template <typename T> class NaiveErf : public NativeUnary<T> {
        T doCompute(T val) const override { return std::erf(val); }
    };

    template <typename T> class NaiveACos : public NativeUnary<T> {
        T doCompute(T val) const override { return std::acos(val); }
    };

    template <typename T> class NaiveACosh : public NativeUnary<T> {
        T doCompute(T val) const override { return std::acosh(val); }
    };

    template <typename T> class NaiveASin : public NativeUnary<T> {
        T doCompute(T val) const override { return std::asin(val); }
    };

    template <typename T> class NaiveASinh : public NativeUnary<T> {
        T doCompute(T val) const override { return std::asinh(val); }
    };

    template <typename T> class NaiveATanh : public NativeUnary<T> {
        T doCompute(T val) const override { return std::atanh(val); }

        template <typename T> class NaiveNeg : public NativeUnary<T> {
            T doCompute(T val) const override { return -val; }
        };

        template <typename T> class Clip : public CpuKernelWithoutConfig {
            void compute(const Operator &_op,
                         const RuntimeObj *context) const override {
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
        };

        template <typename T> class Log : public CpuKernelWithoutConfig {
            void compute(const Operator &_op,
                         const RuntimeObj *context) const override {
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
        };

        template <typename T> class NaiveATan : public NativeUnary<T> {
            T doCompute(T val) const override { return std::atan(val); }
        };

        REGISTER_KERNEL(Device::CPU, OpType::Relu, DataType::UInt32,
                        NaiveRelu<uint32_t>, "reluNaive_CPU_uint32");
        REGISTER_KERNEL(Device::CPU, OpType::Relu, DataType::Float32,
                        NaiveRelu<float>, "reluNaive_CPU_float32");
        REGISTER_KERNEL(Device::CPU, OpType::Gelu, DataType::UInt32,
                        NaiveGelu<float>, "geluNaive_CPU_float32");
        REGISTER_KERNEL(Device::CPU, OpType::Gelu, DataType::Float32,
                        NaiveGelu<float>, "geluNaive_CPU_float32");
        REGISTER_KERNEL(Device::CPU, OpType::Sigmoid, DataType::UInt32,
                        NaiveSigmoid<uint32_t>, "sigmoidNaive_CPU_uint32");
        REGISTER_KERNEL(Device::CPU, OpType::Sigmoid, DataType::Float32,
                        NaiveSigmoid<float>, "sigmoidNaive_CPU_float32");
        REGISTER_KERNEL(Device::CPU, OpType::HardSigmoid, DataType::Float32,
                        NaiveHardSigmoid<float>,
                        "hardSigmoidNaive_CPU_float32");
        REGISTER_KERNEL(Device::CPU, OpType::HardSwish, DataType::Float32,
                        NaiveHardSwish<float>, "hardSwishNaive_CPU_float32");
        REGISTER_KERNEL(Device::CPU, OpType::Tanh, DataType::UInt32,
                        NaiveTanh<uint32_t>, "tanhNaive_CPU_uint32");
        REGISTER_KERNEL(Device::CPU, OpType::Tanh, DataType::Float32,
                        NaiveTanh<float>, "tanhNaive_CPU_float32");
        REGISTER_KERNEL(Device::CPU, OpType::Abs, DataType::UInt32,
                        NaiveAbs<uint32_t>, "absNaive_CPU_uint32");
        REGISTER_KERNEL(Device::CPU, OpType::Abs, DataType::Float32,
                        NaiveAbs<float>, "absNaive_CPU_float32");
        REGISTER_KERNEL(Device::CPU, OpType::Sqrt, DataType::Float32,
                        NaiveSqrt<float>, "sqrtNaive_CPU_float32");
        REGISTER_KERNEL(Device::CPU, OpType::Erf, DataType::Float32,
                        NaiveErf<float>, "erfNaive_CPU_float32");
        REGISTER_KERNEL(Device::CPU, OpType::Neg, DataType::Float32,
                        NaiveNeg<float>, "negNaive_CPU_float32");
        REGISTER_KERNEL(Device::CPU, OpType::Softmax, DataType::UInt32,
                        NaiveSoftmax<uint32_t>, "softmaxNaive_CPU_uint32");
        REGISTER_KERNEL(Device::CPU, OpType::Softmax, DataType::Float32,
                        NaiveSoftmax<float>, "softmaxNaive_CPU_float32");
        REGISTER_KERNEL(Device::CPU, OpType::Clip, DataType::Float32,
                        Clip<float>, "Clip_CPU_float32");
        REGISTER_KERNEL(Device::CPU, OpType::Atan, DataType::Float32,
                        NaiveATan<float>, "Atan_CPU_float32");
        REGISTER_KERNEL(Device::CPU, OpType::Log, DataType::Float32, Log<float>,
                        "Log_CPU_float32");
        REGISTER_KERNEL(Device::CPU, OpType::Cos, DataType::Float32,
                        NaiveCos<float>, "Cos_CPU_float32");
        REGISTER_KERNEL(Device::CPU, OpType::Sin, DataType::Float32,
                        NaiveSin<float>, "Sin_CPU_float32");
        REGISTER_KERNEL(Device::CPU, OpType::Tan, DataType::Float32,
                        NaiveTan<float>, "Tan_CPU_float32");
        REGISTER_KERNEL(Device::CPU, OpType::Sinh, DataType::Float32,
                        NaiveSinh<float>, "Sinh_CPU_float32");
        REGISTER_KERNEL(Device::CPU, OpType::Cosh, DataType::Float32,
                        NaiveCosh<float>, "Cosh_CPU_float32");
        REGISTER_KERNEL(Device::CPU, OpType::Erf, DataType::Float32,
                        NaiveErf<float>, "Erf_CPU_float32");
        REGISTER_KERNEL(Device::CPU, OpType::Acos, DataType::Float32,
                        NaiveACos<float>, "ACos_CPU_float32");
        REGISTER_KERNEL(Device::CPU, OpType::Acosh, DataType::Float32,
                        NaiveACosh<float>, "ACosh_CPU_float32");
        REGISTER_KERNEL(Device::CPU, OpType::Asin, DataType::Float32,
                        NaiveASin<float>, "ASin_CPU_float32");
        REGISTER_KERNEL(Device::CPU, OpType::Asinh, DataType::Float32,
                        NaiveASinh<float>, "ASinh_CPU_float32");
        REGISTER_KERNEL(Device::CPU, OpType::Atanh, DataType::Float32,
                        NaiveATanh<float>, "ATanh_CPU_float32");
    }; // namespace infini

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
template <typename T> class NaiveTanh : public NativeUnary<T> {
    T doCompute(T val) const override {
        return (pow(E_CONSTANT, val) - pow(E_CONSTANT, -val)) /
               (pow(E_CONSTANT, val) + pow(E_CONSTANT, -val));
    }
};
template <typename T> class NaiveAbs : public NativeUnary<T> {
    T doCompute(T val) const override { return val < 0 ? -val : val; }
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

REGISTER_KERNEL(Device::CPU, OpType::Relu, DataType::UInt32,
                NaiveRelu<uint32_t>, "reluNaive_CPU_uint32");
REGISTER_KERNEL(Device::CPU, OpType::Relu, DataType::Float32, NaiveRelu<float>,
                "reluNaive_CPU_float32");
REGISTER_KERNEL(Device::CPU, OpType::Sigmoid, DataType::UInt32,
                NaiveSigmoid<uint32_t>, "sigmoidNaive_CPU_uint32");
REGISTER_KERNEL(Device::CPU, OpType::Sigmoid, DataType::Float32,
                NaiveSigmoid<float>, "sigmoidNaive_CPU_float32");
REGISTER_KERNEL(Device::CPU, OpType::Tanh, DataType::UInt32,
                NaiveTanh<uint32_t>, "tanhNaive_CPU_uint32");
REGISTER_KERNEL(Device::CPU, OpType::Tanh, DataType::Float32, NaiveTanh<float>,
                "tanhNaive_CPU_float32");
REGISTER_KERNEL(Device::CPU, OpType::Abs, DataType::UInt32, NaiveAbs<uint32_t>,
                "absNaive_CPU_uint32");
REGISTER_KERNEL(Device::CPU, OpType::Abs, DataType::Float32, NaiveAbs<float>,
                "absNaive_CPU_float32");
REGISTER_KERNEL(Device::CPU, OpType::Softmax, DataType::UInt32,
                NaiveSoftmax<uint32_t>, "softmaxNaive_CPU_uint32");
REGISTER_KERNEL(Device::CPU, OpType::Softmax, DataType::Float32,
                NaiveSoftmax<float>, "softmaxNaive_CPU_float32");
REGISTER_KERNEL(Device::CPU, OpType::Clip, DataType::Float32, Clip<float>,
                "Clip_CPU_float32");
}; // namespace infini

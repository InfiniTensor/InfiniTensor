#include "operators/element_wise.h"
#include "core/kernel.h"

namespace infini {
template <typename T> class NativeElementWise : public CpuKernelWithoutConfig {
    virtual T doCompute(T val0, T val1) const = 0;
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<ElementWiseObj>(_op);
        T *inptr0 = op->getInputs(0)->getRawDataPtr<T *>();
        T *inptr1 = op->getInputs(1)->getRawDataPtr<T *>();
        T *outptr = op->getOutput()->getRawDataPtr<T *>();

        int a[4] = {1, 1, 1, 1};
        int b[4] = {1, 1, 1, 1};
        int c[4] = {1, 1, 1, 1};
        auto a_input = op->getInputs(0)->getDims();
        auto b_input = op->getInputs(1)->getDims();
        auto c_output = op->getOutput()->getDims();
        std::copy(a_input.begin(), a_input.end(), a + (4 - a_input.size()));
        std::copy(b_input.begin(), b_input.end(), b + (4 - b_input.size()));
        std::copy(c_output.begin(), c_output.end(), c + (4 - c_output.size()));

        auto n = op->getOutput()->size();
        for (size_t i = 0; i < n; ++i) {
            int c0_index = i / (c[1] * c[2] * c[3]);
            int c1_index = (i % (c[1] * c[2] * c[3])) / (c[2] * c[3]);
            int c2_index = ((i % (c[1] * c[2] * c[3])) % (c[2] * c[3])) / c[3];
            int c3_index = ((i % (c[1] * c[2] * c[3])) % (c[2] * c[3])) % c[3];

            int a0_index = c0_index % a[0];
            int a1_index = c1_index % a[1];
            int a2_index = c2_index % a[2];
            int a3_index = c3_index % a[3];

            int b0_index = c0_index % b[0];
            int b1_index = c1_index % b[1];
            int b2_index = c2_index % b[2];
            int b3_index = c3_index % b[3];
            outptr[i] = doCompute(
                inptr0[a0_index * a[1] * a[2] * a[3] + a1_index * a[2] * a[3] +
                       a2_index * a[3] + a3_index],
                inptr1[b0_index * b[1] * b[2] * b[3] + b1_index * b[2] * b[3] +
                       b2_index * b[3] + b3_index]);
        }
    }
};

template <typename T> class NaiveAdd : public NativeElementWise<T> {
    T doCompute(T val0, T val1) const override { return val0 + val1; }
};
template <typename T> class NaiveSub : public NativeElementWise<T> {
    T doCompute(T val0, T val1) const override { return val0 - val1; }
};
template <typename T> class NaiveMul : public NativeElementWise<T> {
    T doCompute(T val0, T val1) const override { return val0 * val1; }
};
template <typename T> class NaiveDiv : public NativeElementWise<T> {
    T doCompute(T val0, T val1) const override { return (T)(val0 / val1); }
};

REGISTER_KERNEL(Device::CPU, OpType::Add, DataType::UInt32, NaiveAdd<uint32_t>,
                "addNaive_CPU_uint32");
REGISTER_KERNEL(Device::CPU, OpType::Add, DataType::Float32, NaiveAdd<float>,
                "addNaive_CPU_float32");
REGISTER_KERNEL(Device::CPU, OpType::Sub, DataType::UInt32, NaiveSub<uint32_t>,
                "subNaive_CPU_uint32");
REGISTER_KERNEL(Device::CPU, OpType::Sub, DataType::Float32, NaiveSub<float>,
                "subNaive_CPU_float32");
REGISTER_KERNEL(Device::CPU, OpType::Mul, DataType::UInt32, NaiveMul<uint32_t>,
                "mulNaive_CPU_uint32");
REGISTER_KERNEL(Device::CPU, OpType::Mul, DataType::Float32, NaiveMul<float>,
                "mulNaive_CPU_float32");
REGISTER_KERNEL(Device::CPU, OpType::Div, DataType::UInt32, NaiveDiv<uint32_t>,
                "divNaive_CPU_uint32");
REGISTER_KERNEL(Device::CPU, OpType::Div, DataType::Float32, NaiveDiv<float>,
                "divNaive_CPU_float32");
}; // namespace infini

#include "operators/element_wise.h"
#include "core/kernel.h"

namespace infini {
class NativeElementWise : public CpuKernelWithoutConfig {
    template <typename T> static T addCompute(T val0, T val1) {
        return val0 + val1;
    }

    template <typename T> static T subCompute(T val0, T val1) {
        return val0 - val1;
    }

    template <typename T> static T mulCompute(T val0, T val1) {
        return val0 * val1;
    }

    template <typename T> static T divCompute(T val0, T val1) {
        return (T)(val0 / val1);
    }

    template <typename T> static T equalCompute(T val0, T val1) {
        return (T)(val0 == val1);
    }

    template <typename T> static T greaterOrEqualCompute(T val0, T val1) {
        return (T)(val0 >= val1);
    }

    template <typename T> static T greaterCompute(T val0, T val1) {
        return (T)(val0 > val1);
    }

    template <typename T> static T lessOrEqualCompute(T val0, T val1) {
        return (T)(val0 <= val1);
    }

    template <typename T> static T lessCompute(T val0, T val1) {
        return (T)(val0 < val1);
    }

    template <typename T>
    void doCompute(const Operator &_op, const RuntimeObj *context) const {
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
        T (*_doCompute)(T val0, T val1);
        switch (op->getOpType().underlying()) {
        case OpType::Add:
            _doCompute = addCompute<T>;
            break;
        case OpType::Sub:
            _doCompute = subCompute<T>;
            break;
        case OpType::Mul:
            _doCompute = mulCompute<T>;
            break;
        case OpType::Div:
            _doCompute = divCompute<T>;
            break;
        case OpType::Equal:
            _doCompute = equalCompute<T>;
            break;
        case OpType::GreaterOrEqual:
            _doCompute = greaterOrEqualCompute<T>;
            break;
        case OpType::Greater:
            _doCompute = greaterCompute<T>;
            break;
        case OpType::LessOrEqual:
            _doCompute = lessOrEqualCompute<T>;
            break;
        case OpType::Less:
            _doCompute = lessCompute<T>;
            break;
        default:
            IT_TODO_HALT();
        }

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
            outptr[i] = _doCompute(
                inptr0[a0_index * a[1] * a[2] * a[3] + a1_index * a[2] * a[3] +
                       a2_index * a[3] + a3_index],
                inptr1[b0_index * b[1] * b[2] * b[3] + b1_index * b[2] * b[3] +
                       b2_index * b[3] + b3_index]);
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

REGISTER_KERNEL(Device::CPU, OpType::Add, NativeElementWise, "addNaive_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Sub, NativeElementWise, "subNaive_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Mul, NativeElementWise, "mulNaive_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Div, NativeElementWise, "divNaive_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Equal, NativeElementWise,
                "equalNaive_CPU");
REGISTER_KERNEL(Device::CPU, OpType::GreaterOrEqual, NativeElementWise,
                "greaterEqualNaive_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Greater, NativeElementWise,
                "greaterThanNaive_CPU");
REGISTER_KERNEL(Device::CPU, OpType::LessOrEqual, NativeElementWise,
                "lessEqualNaive_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Less, NativeElementWise,
                "lessEqualNaive_CPU");
}; // namespace infini

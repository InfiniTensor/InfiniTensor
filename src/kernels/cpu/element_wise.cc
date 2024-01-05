#include "operators/element_wise.h"
#include "core/kernel.h"
#include "utils/operator_utils.h"

namespace infini {
template <typename T> class NativeElementWise : public CpuKernelWithoutConfig {
    virtual T doCompute(T val0, T val1) const = 0;
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<ElementWiseObj>(_op);
        T *inptr0 = op->getInputs(0)->getRawDataPtr<T *>();
        T *inptr1 = op->getInputs(1)->getRawDataPtr<T *>();
        T *outptr = op->getOutput()->getRawDataPtr<T *>();

        auto shapeA = op->getInputs(0)->getDims();
        auto shapeB = op->getInputs(1)->getDims();
        auto shapeC = op->getOutput()->getDims();
        auto rank = op->getOutput()->getRank();
        Shape a(rank, 1);
        Shape b(rank, 1);
        std::copy(shapeA.begin(), shapeA.end(),
                  a.begin() + (rank - shapeA.size()));
        std::copy(shapeB.begin(), shapeB.end(),
                  b.begin() + (rank - shapeB.size()));
        auto getStride = [&](const Shape &shape) {
            int p = 1;
            Shape stride(rank);
            for (auto i = rank; i > 0; --i) {
                stride[i - 1] = p;
                p = p * shape[i - 1];
            }
            return stride;
        };
        Shape strideA = getStride(a);
        Shape strideB = getStride(b);

        auto n = op->getOutput()->size();
        for (size_t i = 0; i < n; ++i) {
            auto shapeIndexC = locate_index(i, shapeC);
            auto indexA = delocate_index(shapeIndexC, a, strideA);
            auto indexB = delocate_index(shapeIndexC, b, strideB);
            outptr[i] = doCompute(inptr0[indexA], inptr1[indexB]);
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
template <typename T> class NaiveEqual : public NativeElementWise<T> {
    T doCompute(T val0, T val1) const override { return (T)(val0 == val1); }
};
template <typename T> class NaiveGreaterEqual : public NativeElementWise<T> {
    T doCompute(T val0, T val1) const override { return (T)(val0 >= val1); }
};
template <typename T> class NaiveGreaterThan : public NativeElementWise<T> {
    T doCompute(T val0, T val1) const override { return (T)(val0 > val1); }
};
template <typename T> class NaiveLessEqual : public NativeElementWise<T> {
    T doCompute(T val0, T val1) const override { return (T)(val0 <= val1); }
};
template <typename T> class NaiveLessThan : public NativeElementWise<T> {
    T doCompute(T val0, T val1) const override { return (T)(val0 < val1); }
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
REGISTER_KERNEL(Device::CPU, OpType::Equal, DataType::UInt32,
                NaiveEqual<uint32_t>, "equalNaive_CPU_uint32");
REGISTER_KERNEL(Device::CPU, OpType::Equal, DataType::Float32,
                NaiveEqual<float>, "equalNaive_CPU_float32");
REGISTER_KERNEL(Device::CPU, OpType::GreaterOrEqual, DataType::UInt32,
                NaiveGreaterEqual<uint32_t>, "greaterEqualNaive_CPU_uint32");
REGISTER_KERNEL(Device::CPU, OpType::GreaterOrEqual, DataType::Float32,
                NaiveGreaterEqual<float>, "greaterEqualNaive_CPU_float32");
REGISTER_KERNEL(Device::CPU, OpType::Greater, DataType::UInt32,
                NaiveGreaterThan<uint32_t>, "greaterThanNaive_CPU_uint32");
REGISTER_KERNEL(Device::CPU, OpType::Greater, DataType::Float32,
                NaiveGreaterThan<float>, "greaterThanNaive_CPU_float32");
REGISTER_KERNEL(Device::CPU, OpType::LessOrEqual, DataType::UInt32,
                NaiveLessEqual<uint32_t>, "lessEqualNaive_CPU_uint32");
REGISTER_KERNEL(Device::CPU, OpType::LessOrEqual, DataType::Float32,
                NaiveLessEqual<float>, "lessEqualNaive_CPU_float32");
REGISTER_KERNEL(Device::CPU, OpType::Less, DataType::UInt32,
                NaiveLessThan<uint32_t>, "lessEqualNaive_CPU_uint32");
REGISTER_KERNEL(Device::CPU, OpType::Less, DataType::Float32,
                NaiveLessThan<float>, "lessEqualNaive_CPU_float32");
}; // namespace infini

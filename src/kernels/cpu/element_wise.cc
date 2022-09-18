#include "operators/element_wise.h"
#include "core/kernel.h"

namespace infini {
template <typename T> class NativeElementWise : public Kernel {
    virtual T doCompute(T val0, T val1) const = 0;
    void compute(const Operator &_op, const PerfRecord &record,
                 const RuntimeObj *context) const override {
        auto op = as<ElementWiseObj>(_op);
        T *inptr0 = op->getInputs(0)->getRawDataPtr<T *>();
        T *inptr1 = op->getInputs(1)->getRawDataPtr<T *>();
        T *outptr = op->getOutput()->getRawDataPtr<T *>();

        auto outDim = op->getOutput()->getDims();
        auto n = op->getOutput()->size();
        for (size_t offset = 0; offset < n; offset++) {
            // For now,we only process the same dims here, broardcast will be
            // considered in the opt layer.
            /*auto offset0 =
                op->getInputs(0)->getOffsetByBroadcastOffset(offset, outDim);
            auto offset1 =
                op->getInputs(1)->getOffsetByBroadcastOffset(offset, outDim);
            outptr[offset] = doCompute(inptr0[offset0], inptr1[offset1]);*/
            outptr[offset] = doCompute(inptr0[offset], inptr1[offset]);
        }
    }

    void compute(const Operator &op, const RuntimeObj *context) const override {
        compute(op, {}, context);
    }

    PerfRecord tune(const Operator &op,
                    const RuntimeObj *context) const override {
        return make_ref<PerfRecordObj>(timeit([&]() { compute(op, context); }));
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
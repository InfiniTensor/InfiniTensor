#include "operators/element_wise.h"
#include "intelcpu/mkl_kernel_without_config.h"
#include "intelcpu/mkl_runtime.h"
#include "operators/unary.h"

namespace infini {
class MklBinary : public MklKernelWithoutConfig {
    dnnl::algorithm getAlgorithem(const Ref<ElementWiseObj> &op) const {
        switch (op->getOpType().underlying()) {
        case OpType::Add:
            return dnnl::algorithm::binary_add;
        case OpType::Sub:
            return dnnl::algorithm::binary_sub;
        case OpType::Mul:
            return dnnl::algorithm::binary_mul;
        case OpType::Div:
            return dnnl::algorithm::binary_div;

        default:
            IT_TODO_HALT();
        }
        return dnnl::algorithm::undef;
    }

    // Binary primitives support elementwise broadcast
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const MklRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        //  create user memory that describes data layout in the buffers
        std::vector<dnnl_dim_t> dims;
        for (size_t i = 0; i < op->getInputs(0)->getRank(); ++i)
            dims.push_back(op->getInputs(0)->getDims()[i]);

        auto srcMd1 = dnnl::memory::desc(dims, dnnl::memory::data_type::f32,
                                         getUserFormatTag(dims.size()));
        auto srcMemory1 = dnnl::memory(srcMd1, context->getEngine(), aData);

        auto srcMd2 = dnnl::memory::desc(dims, dnnl::memory::data_type::f32,
                                         getUserFormatTag(dims.size()));
        auto srcMemory2 = dnnl::memory(srcMd2, context->getEngine(), bData);

        auto dstMd = dnnl::memory::desc(dims, dnnl::memory::data_type::f32,
                                        getUserFormatTag(dims.size()));
        auto output = dnnl::memory(dstMd, context->getEngine(), cData);

        auto binaryDesc =
            dnnl::binary::desc(getAlgorithem(op), srcMd1, srcMd2, dstMd);
        auto primDesc =
            dnnl::binary::primitive_desc(binaryDesc, context->getEngine());

        // create and execute binary primitive
        dnnl::binary(primDesc).execute(context->getStream(),
                                       {{DNNL_ARG_SRC_0, srcMemory1},
                                        {DNNL_ARG_SRC_1, srcMemory2},
                                        {DNNL_ARG_DST, output}});
    }
};

class MklUnary : public MklKernelWithoutConfig {
    dnnl::algorithm getAlgorithem(const Ref<UnaryObj> &op) const {
        switch (op->getOpType().underlying()) {
        case OpType::Relu:
            return dnnl::algorithm::eltwise_relu;
        case OpType::Tanh:
            return dnnl::algorithm::eltwise_tanh;
        case OpType::Abs:
            return dnnl::algorithm::eltwise_abs;
        case OpType::Sigmoid:
            return dnnl::algorithm::eltwise_logistic;
        default:
            IT_TODO_HALT();
        }
        return dnnl::algorithm::undef;
    }

    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const MklRuntimeObj *>(_context);

        void *const srcData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const dstData = (op->getOutput()->getRawDataPtr<void *>());

        //  create user memory that describes data layout in the buffers
        std::vector<dnnl_dim_t> dims;
        for (size_t i = 0; i < op->getInputs(0)->getRank(); ++i)
            dims.push_back(op->getInputs(0)->getDims()[i]);

        auto srcMd = dnnl::memory::desc(dims, dnnl::memory::data_type::f32,
                                        getUserFormatTag(dims.size()), false);
        auto srcMemory = dnnl::memory(srcMd, context->getEngine(), srcData);

        auto output = dnnl::memory(srcMd, context->getEngine(), dstData);

        const float negative1_slope = 0.0f;

        auto unaryDesc = dnnl::eltwise_forward::desc(
            dnnl::prop_kind::forward_inference, getAlgorithem(op), srcMd,
            negative1_slope);
        auto primDesc = dnnl::eltwise_forward::primitive_desc(
            unaryDesc, context->getEngine());

        // create and execute binary primitive
        dnnl::eltwise_forward(primDesc).execute(
            context->getStream(),
            {{DNNL_ARG_SRC, srcMemory}, {DNNL_ARG_DST, output}});
    }
};

REGISTER_KERNEL(Device::INTELCPU, OpType::Add, MklBinary, "Add_Mkl");
REGISTER_KERNEL(Device::INTELCPU, OpType::Sub, MklBinary, "Sub_Mkl");
REGISTER_KERNEL(Device::INTELCPU, OpType::Mul, MklBinary, "Mul_Mkl");
REGISTER_KERNEL(Device::INTELCPU, OpType::Div, MklBinary, "Div_Mkl");

REGISTER_KERNEL(Device::INTELCPU, OpType::Relu, MklUnary, "Relu_Mkl");
REGISTER_KERNEL(Device::INTELCPU, OpType::Sigmoid, MklUnary, "Sigmoid_Mkl");
REGISTER_KERNEL(Device::INTELCPU, OpType::Tanh, MklUnary, "Tanh_Mkl");
REGISTER_KERNEL(Device::INTELCPU, OpType::Abs, MklUnary, "Abs_Mkl");
} // namespace infini

#include "operators/element_wise.h"
#include "bang/bang_kernel_wihtout_config.h"
#include "bang/bang_runtime.h"

namespace infini {
class ElementWiseCnnl : public Kernel {
    virtual cnnlOpTensorDesc_t getOpType() const = 0;
    virtual tuple<float, float, float> getAlphBeta() const {
        return {1.f, 1.f, 0.f};
    }
    void compute(const Operator &_op, const PerfRecord &record,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        cnnlTensorDescriptor_t aDesc, bDesc, cDesc;
        auto dim = op->getInputs(0)->getDims();
        if (dim.size() != 4)
            IT_TODO_HALT();

        int dim_array[4] = {dim[0], dim[1], dim[2], dim[3]};
        // get inputs
        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            aDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_FLOAT, 4, dim_array));

        checkCnnlError(cnnlCreateTensorDescriptor(&bDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            bDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_FLOAT, 4, dim_array));

        // get outputs
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            cDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_FLOAT, 4, dim_array));

        // get op descriptor
        cnnlOpTensorDescriptor_t opDesc;
        checkCnnlError(cnnlCreateOpTensorDescriptor(&opDesc));
        checkCnnlError(cnnlSetOpTensorDescriptor(
            opDesc, getOpType(), CNNL_DTYPE_FLOAT, CNNL_NOT_PROPAGATE_NAN));

        size_t wsSize;
        cnnlGetOpTensorWorkspaceSize(context->cnnlHandle(), aDesc, bDesc, cDesc, &wsSize);

        BangPtr wsData = context->getWorkspace(wsSize);

        auto [aAlpha, bAlpha, beta] = getAlphBeta();
        cnnlStatus_t stat =
            cnnlOpTensor(context->cnnlHandle(), opDesc, &aAlpha, aDesc, aData,
                          &bAlpha, bDesc, bData, wsData, wsSize, &beta, cDesc, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        // Destories in BANG does not require sync. But cnnl does not state
        // whether sync is required before destories.
        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(bDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
        checkCnnlError(cnnlDestroyOpTensorDescriptor(opDesc));
    }

    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        compute(_op, {}, _context);
    }
    // Premise: op is idempotent since it is called multiple times.
    PerfRecord tune(const Operator &_op,
                    const RuntimeObj *_context) const override {
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);
        return make_ref<PerfRecordObj>(timeit([&]() { compute(_op, _context); },
                                              [&]() { context->sync(); }));
    }
};

class AddCnnl : public ElementWiseCnnl {
    cnnlOpTensorDesc_t getOpType() const override { return CNNL_OP_TENSOR_ADD; }
};

class SubCnnl : public ElementWiseCnnl {
    cnnlOpTensorDesc_t getOpType() const override { return CNNL_OP_TENSOR_ADD; }
    tuple<float, float, float> getAlphBeta() const override {
        return {1.f, -1.f, 0.f};
    }
};

class MulCnnl : public ElementWiseCnnl {
    cnnlOpTensorDesc_t getOpType() const override { return CNNL_OP_TENSOR_MUL; }
};

// class ElementWiseBang : public BangKernelWithoutConfig {
//     void compute(const Operator &_op,
//                  const RuntimeObj *_context) const override {
//         element_wise_kernel(_op);
//     }
// };

REGISTER_KERNEL(Device::BANG, OpType::Add, DataType::Float32, AddCnnl,
                "Add_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::Sub, DataType::Float32, SubCnnl,
                "Sub_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::Mul, DataType::Float32, MulCnnl,
                "Mul_cnnl_BANG_Float32");

// REGISTER_KERNEL(Device::BANG, OpType::Div, DataType::Float32, ElementWiseBang,
//                 "Div_Bang_Float32");
// REGISTER_KERNEL(Device::BANG, OpType::Pow, DataType::Float32, ElementWiseBang,
//                 "Pow_Bang_Float32");
}; // namespace infini

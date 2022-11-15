#include "operators/matmul.h"
#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"

namespace infini {
class MatmulCnnl : public BangKernelWithoutConfig {
    virtual tuple<float, float> getAlphBeta() const {
        return {1.f, 0.f};
    }
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<MatmulObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        cnnlTensorDescriptor_t aDesc, bDesc, cDesc;
        auto dimInputs0 = op->getInputs(0)->getDims();
        auto dimInputs1 = op->getInputs(1)->getDims();
        auto dimOutput = op->getOutput()->getDims();
        if (dimInputs0.size() != 2)
            IT_TODO_HALT();
        if (dimInputs1.size() != 2)
            IT_TODO_HALT();
        if (dimOutput.size() != 2)
            IT_TODO_HALT();

        bool transA = op->getTransA();
        bool transB = op->getTransB();

        int inputs0Array[2] = {dimInputs0[0], dimInputs0[1]};
        int inputs1Array[2] = {dimInputs1[0], dimInputs1[1]};
        int outputArray[2] = {dimOutput[0], dimOutput[1]};

        // get inputs
        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_ARRAY,
                                               CNNL_DTYPE_FLOAT, 2, inputs0Array));

        checkCnnlError(cnnlCreateTensorDescriptor(&bDesc));
        checkCnnlError(cnnlSetTensorDescriptor(bDesc, CNNL_LAYOUT_ARRAY,
                                               CNNL_DTYPE_FLOAT, 2, inputs1Array));

        // get outputs
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_ARRAY,
                                               CNNL_DTYPE_FLOAT, 2, outputArray));


        auto [alpha, beta] = getAlphBeta();
        cnnlStatus_t stat = cnnlMatMul(context->cnnlHandle(), transA, transB, &alpha,
                                         aDesc, aData, bDesc, bData, &beta, cDesc, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        // Destories in BANG does not require sync. But cnnl does not state
        // whether sync is required before destories.
        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(bDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
    }
};

// class AddCnnl : public MatmulCnnl {
//     cnnlOpTensorDesc_t getOpType() const override { return CNNL_OP_TENSOR_ADD; }
// };

REGISTER_KERNEL(Device::BANG, OpType::Matmul, DataType::Float32, MatmulCnnl,
                "Matmul_cnnl_BANG_Float32");
}; // namespace infini

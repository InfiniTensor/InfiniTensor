#include "operators/matmul.h"
#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"

namespace infini {
class MatmulCnnl : public BangKernelWithoutConfig {
    virtual tuple<float, float> getAlphBeta() const { return {1.f, 0.f}; }
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
        int input0_batch_size = 1;
        int input1_batch_size = 1;
        int output_batch_size = 1;
        for (size_t i = 0; i < dimInputs0.size() - 2; ++i) {
            input0_batch_size *= dimInputs0[i];
            input1_batch_size *= dimInputs1[i];
            output_batch_size *= dimOutput[i];
        }

        bool transA = op->getTransA();
        bool transB = op->getTransB();

        int inputs0Array[3] = {input0_batch_size,
                               dimInputs0[dimInputs0.size() - 2],
                               dimInputs0[dimInputs0.size() - 1]};
        int inputs1Array[3] = {input1_batch_size,
                               dimInputs1[dimInputs1.size() - 2],
                               dimInputs1[dimInputs1.size() - 1]};
        int outputArray[3] = {output_batch_size,
                              dimOutput[dimOutput.size() - 2],
                              dimOutput[dimOutput.size() - 1]};

        // get inputs
        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            aDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 3, inputs0Array));

        checkCnnlError(cnnlCreateTensorDescriptor(&bDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            bDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 3, inputs1Array));

        // get outputs
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            cDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 3, outputArray));

        cnnlStatus_t stat =
            cnnlBatchMatMul(context->cnnlHandle(), transA, transB, aDesc, aData,
                            bDesc, bData, cDesc, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        // Destories in BANG does not require sync. But cnnl does not state
        // whether sync is required before destories.
        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(bDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::Matmul, DataType::Float32, MatmulCnnl,
                "Matmul_cnnl_BANG_Float32");
}; // namespace infini

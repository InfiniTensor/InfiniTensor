#include "operators/transpose.h"
#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"

namespace infini {
class TransposeCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<TransposeObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        cnnlTensorDescriptor_t aDesc, cDesc;
        auto dimin = op->getInputs(0)->getDims();
        auto dimout = op->getOutput()->getDims();
        if (dimin.size() != 4 || dimout.size() != 4)
            IT_TODO_HALT();

        int dimin_array[4] = {dimin[0], dimin[1], dimin[2], dimin[3]};
        int dimout_array[4] = {dimout[0], dimout[1], dimout[2], dimout[3]};
        // get inputs
        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            aDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 4, dimin_array));

        // get outputs
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            cDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 4, dimout_array));

        // get op descriptor
        auto permute = op->getPermute();
        cnnlTransposeDescriptor_t opDesc;
        checkCnnlError(cnnlCreateTransposeDescriptor(&opDesc));
        checkCnnlError(cnnlSetTransposeDescriptor(opDesc, 4, permute.data()));

        size_t wsSize;
        cnnlGetTransposeWorkspaceSize(context->cnnlHandle(), aDesc, opDesc,
                                      &wsSize);
        BangPtr wsData = context->getWorkspace(wsSize);

        cnnlStatus_t stat =
            cnnlTranspose_v2(context->cnnlHandle(), opDesc, aDesc, aData, cDesc,
                             cData, wsData, wsSize);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        // Destories in BANG does not require sync. But cnnl does not state
        // whether sync is required before destories.
        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
        checkCnnlError(cnnlDestroyTransposeDescriptor(opDesc));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::Transpose, DataType::Float32,
                TransposeCnnl, "Transpose_cnnl_BANG_Float32");
}; // namespace infini

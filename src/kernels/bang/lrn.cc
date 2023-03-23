#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"
#include "operators/unary.h"

namespace infini {
class LrnCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<LrnObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        int lrn_n = op->getFeatureNum();
        float alpha = op->getAlpha();
        float beta = op->getBeta();
        float bias = op->getBias();

        cnnlTensorDescriptor_t aDes, cDesc;
        auto dim = op->getOutput()->getDims();
        float alpha = op->getAlpha();
        float beta = op->getBeta();
        if (dim.size() != 4)
            IT_TODO_HALT();

        int dim_array[4] = {dim[0], dim[1], dim[2], dim[3]};
        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));

        size_t wsSize;
        cnnlGetLrnWorkspaceSize_v2(context->cnnlHandle(), aDesc, cDesc, CNNL_LRN_CROSS_CHANNEL, lrn_n, 
                                     &wsSize);

        BangPtr wsData = context->getWorkspace(wsSize);

        cnnlStatus_t stat = cnnlLrn(context->cnnlHandle(), CNNL_LRN_CROSS_CHANNEL, lrn_n, double(alpha),
                                    double(beta), double(bias), wsData, wsSize, aDesc,
                                          aData, cDesc, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        // Destories in BANG does not require sync. But cnnl does not state
        // whether sync is required before destories.
        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::Lrn, DataType::Float32,
                LrnCnnl, "Lrn_cnnl_BANG_Float32");

}; // namespace infini

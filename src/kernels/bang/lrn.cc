#include "operators/lrn.h"
#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"

namespace infini {
class LRNCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<LRNObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        cnnlTensorDescriptor_t aDesc, cDesc;
        auto aDim = op->getInputs(0)->getDims();
        auto cDim = op->getOutput()->getDims();
        auto [alpha, beta, bias] = op->getAlphaBetaBias();
        auto size = op->getSize();

        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, aDim.size(),
                                               aDim.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, cDim.size(),
                                               cDim.data()));

        size_t extra_size;
        cnnlGetLrnExtraInputSize_v2(context->cnnlHandle(), cDesc,
                                    CNNL_LRN_LOCAL_SIZE, size, &extra_size);
        void *extra_cpu = NULL;
        extra_cpu = malloc(extra_size);
        BangPtr extra_mlu = context->getWorkspace(extra_size);
        cnnlInitLrnExtraInput(context->cnnlHandle(), CNNL_LRN_LOCAL_SIZE, size,
                              (double)alpha, (double)beta, (double)bias, aDesc,
                              cDesc, extra_cpu);
        cnrtMemcpy(extra_mlu, extra_cpu, extra_size,
                   CNRT_MEM_TRANS_DIR_HOST2DEV);

        size_t wsSize;
        cnnlGetLrnWorkspaceSize_v2(context->cnnlHandle(), aDesc, cDesc,
                                   CNNL_LRN_LOCAL_SIZE, size, &wsSize);
        BangPtr wsData = context->getWorkspace(wsSize);

        cnnlStatus_t stat = cnnlLrn_v2(
            context->cnnlHandle(), CNNL_LRN_LOCAL_SIZE, size, (double)alpha,
            (double)beta, (double)bias, wsData, wsSize, aDesc, aData, extra_mlu,
            extra_size, cDesc, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::LRN, DataType::Float32, LRNCnnl,
                "LRN_cnnl_BANG_Float32");

}; // namespace infini

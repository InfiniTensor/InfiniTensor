#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"
#include "operators/reduce_mean.h"

namespace infini {
class ReduceMeanCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ReduceMeanObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        cnnlTensorDescriptor_t aDesc, cDesc;

        auto dimIn = op->getInputs(0)->getDims();
        auto dimOut = op->getOutput()->getDims();

        auto axes = op->getAxes();
        vector<int> axes_vector;
        axes_vector.assign(axes.begin(), axes.end());

        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_ARRAY,
                                               CNNL_DTYPE_FLOAT, dimIn.size(), dimIn.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_ARRAY,
                                               CNNL_DTYPE_FLOAT, dimOut.size(), dimOut.data()));
        cnnlReduceDescriptor_t reduceDesc;
        checkCnnlError(cnnlCreateReduceDescriptor(&reduceDesc));
        checkCnnlError(cnnlSetReduceDescriptor(reduceDesc, axes_vector.data(), axes_vector.size(),
                                               CNNL_REDUCE_AVG, CNNL_DTYPE_FLOAT, CNNL_NOT_PROPAGATE_NAN, CNNL_REDUCE_NO_INDICES, CNNL_32BIT_INDICES));

        size_t wsSize;
        cnnlGetReduceOpWorkspaceSize(context->cnnlHandle(), aDesc, cDesc, reduceDesc, &wsSize);
        BangPtr wsData = context->getWorkspace(wsSize);

        void *ptr;
        cnrtMalloc(&ptr, axes_vector.size()*sizeof(int));
        cnrtMemcpy(ptr, axes_vector.data(), axes_vector.size() * sizeof(int), CNRT_MEM_TRANS_DIR_HOST2DEV);

        cnnlStatus_t stat =
            cnnlReduce(context->cnnlHandle(), reduceDesc, wsData, wsSize, NULL, aDesc, aData, axes_vector.size()*sizeof(int), ptr, NULL, cDesc, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        checkBangError(cnrtFree(ptr));
        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
        checkCnnlError(cnnlDestroyReduceDescriptor(reduceDesc));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::ReduceMean, DataType::Float32, ReduceMeanCnnl,
                "Reducemean_cnnl_BANG_Float32");

}; // namespace infini

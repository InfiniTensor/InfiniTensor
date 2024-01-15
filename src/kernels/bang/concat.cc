#include "operators/concat.h"
#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"

namespace infini {
class ConcatCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ConcatObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);
        int num = op->numInputs();
        int axis = op->getDim();

        auto cDim = op->getOutput()->getDims();
        cnnlTensorDescriptor_t desc;
        checkCnnlError(cnnlCreateTensorDescriptor(&desc));
        checkCnnlError(cnnlSetTensorDescriptor(desc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, cDim.size(),
                                               cDim.data()));

        cnnlTensorDescriptor_t descArray[num];
        for (int i = 0; i < num; ++i) {
            checkCnnlError(cnnlCreateTensorDescriptor(&descArray[i]));
            checkCnnlError(cnnlSetTensorDescriptor(
                descArray[i], CNNL_LAYOUT_NCHW, CNNL_DTYPE_FLOAT,
                op->getInputs(i)->getDims().size(),
                op->getInputs(i)->getDims().data()));
        }

        void *argv[num];
        for (int i = 0; i < num; ++i) {
            argv[i] = op->getInputs(i)->getRawDataPtr<void *>();
        }
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        size_t wsSize;
        cnnlGetConcatWorkspaceSize(context->cnnlHandle(), num, &wsSize);
        BangPtr wsData = context->getWorkspace(wsSize);

        cnnlStatus_t stat =
            cnnlConcat(context->cnnlHandle(), num, axis, descArray, argv,
                       wsData, wsSize, desc, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        for (int i = 0; i < num; ++i) {
            checkCnnlError(cnnlDestroyTensorDescriptor(descArray[i]));
        }
        checkCnnlError(cnnlDestroyTensorDescriptor(desc));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::Concat, ConcatCnnl, "Concat_cnnl_BANG");
}; // namespace infini

#include "operators/split.h"
#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"

namespace infini {
class SplitCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<SplitObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);
        int num = op->numOutputs();
        int axis = op->getDim();

        auto dim = op->getInputs(0)->getDims();
        cnnlTensorDescriptor_t desc;
        checkCnnlError(cnnlCreateTensorDescriptor(&desc));
        checkCnnlError(cnnlSetTensorDescriptor(
            desc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_FLOAT, dim.size(), dim.data()));

        cnnlTensorDescriptor_t descArray[num];
        for (int i = 0; i < num; ++i) {
            checkCnnlError(cnnlCreateTensorDescriptor(&descArray[i]));
            checkCnnlError(cnnlSetTensorDescriptor(
                descArray[i], CNNL_LAYOUT_NCHW, CNNL_DTYPE_FLOAT,
                op->getOutput(i)->getDims().size(),
                op->getOutput(i)->getDims().data()));
        }

        void *const inputData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *argv[num];
        for (int i = 0; i < num; ++i) {
            argv[i] = op->getOutput(i)->getRawDataPtr<void *>();
        }

        size_t wsSize;
        cnnlGetSplitWorkspaceSize(context->cnnlHandle(), num, &wsSize);
        BangPtr wsData = context->getWorkspace(wsSize);

        cnnlStatus_t stat =
            cnnlSplit(context->cnnlHandle(), num, axis, desc, inputData, wsData,
                      wsSize, descArray, argv);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        for (int i = 0; i < num; ++i) {
            checkCnnlError(cnnlDestroyTensorDescriptor(descArray[i]));
        }
        checkCnnlError(cnnlDestroyTensorDescriptor(desc));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::Split, SplitCnnl, "Split_cnnl_BANG");
}; // namespace infini

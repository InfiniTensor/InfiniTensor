#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"
#include "operators/concat.h"

namespace infini {
class ConcatCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ConcatObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);
        int num = op->numInputs();
        int axis = op->getDim();
        void *argv[num];
        for (int i = 0; i < num; ++i) {
            argv[i] = op->getInputs(i)->getRawDataPtr<void *>();
        }
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        cnnlTensorDescriptor_t desc;
        auto dim = op->getInputs(0)->getDims();
        if (dim.size() != 4)
            IT_TODO_HALT();

        int dim_array[4] = {dim[0], dim[1], dim[2], dim[3]};
        int dimout_array[4] = {dim[0], dim[1], dim[2], dim[3]};
        dimout_array[axis] *=  num;
        checkCnnlError(cnnlCreateTensorDescriptor(&desc));
        checkCnnlError(cnnlSetTensorDescriptor(desc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dimout_array));
        cnnlTensorDescriptor_t descArray[num];
        for (int i = 0; i < num; ++i) {
            checkCnnlError(cnnlCreateTensorDescriptor(&descArray[i]));
            checkCnnlError(
                cnnlSetTensorDescriptor(descArray[i], CNNL_LAYOUT_NCHW,
                                        CNNL_DTYPE_FLOAT, 4, dim_array));
        }

        size_t wsSize;
        cnnlGetConcatWorkspaceSize(context->cnnlHandle(), num, &wsSize);
        BangPtr wsData = context->getWorkspace(wsSize);

        cnnlStatus_t stat =
            cnnlConcat(context->cnnlHandle(), num, axis, descArray, argv, wsData, wsSize, desc, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        // Destories in BANG does not require sync. But cnnl does not state
        // whether sync is required before destories.
        for (int i = 0; i < num; ++i) {
            checkCnnlError(cnnlDestroyTensorDescriptor(descArray[i]));
        }
        checkCnnlError(cnnlDestroyTensorDescriptor(desc));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::Concat, DataType::Float32, ConcatCnnl,
                "Concat_cnnl_BANG_Float32");
}; // namespace infini

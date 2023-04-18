#include "operators/split.h"
#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"

namespace infini {
class SplitCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<SplitObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);
        int num = op->numOutputs();
        int axis = op->getDim();
        void *argv[num];
        for (int i = 0; i < num; ++i) {
            argv[i] = op->getOutput(i)->getRawDataPtr<void *>();
        }
        void *const inputData = (op->getInputs(0)->getRawDataPtr<void *>());

        cnnlTensorDescriptor_t desc;

        int dimout_array[num][4];
        for (int i = 0; i < num; ++i) {
            auto dim = op->getOutput(i)->getDims();
            if (dim.size() != 4) {
                IT_TODO_HALT();
            }
            dimout_array[i][0] = dim[0];
            dimout_array[i][1] = dim[1];
            dimout_array[i][2] = dim[2];
            dimout_array[i][3] = dim[3];
        }
        auto dim = op->getInputs(0)->getDims();
        if (dim.size() != 4) {
            IT_TODO_HALT();
        }
        int dim_array[4] = {dim[0], dim[1], dim[2], dim[3]};
        checkCnnlError(cnnlCreateTensorDescriptor(&desc));
        checkCnnlError(cnnlSetTensorDescriptor(desc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));
        cnnlTensorDescriptor_t descArray[num];
        for (int i = 0; i < num; ++i) {
            checkCnnlError(cnnlCreateTensorDescriptor(&descArray[i]));
            checkCnnlError(
                cnnlSetTensorDescriptor(descArray[i], CNNL_LAYOUT_NCHW,
                                        CNNL_DTYPE_FLOAT, 4, dimout_array[i]));
        }

        size_t wsSize;
        cnnlGetSplitWorkspaceSize(context->cnnlHandle(), num, &wsSize);
        BangPtr wsData = context->getWorkspace(wsSize);

        cnnlStatus_t stat =
            cnnlSplit(context->cnnlHandle(), num, axis, desc, inputData, wsData,
                      wsSize, descArray, argv);
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

REGISTER_KERNEL(Device::BANG, OpType::Split, DataType::Float32, SplitCnnl,
                "Split_cnnl_BANG_Float32");
}; // namespace infini

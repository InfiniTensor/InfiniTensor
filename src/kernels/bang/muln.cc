#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"
#include "operators/element_wise.h"

namespace infini {
class MulNCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<MulNObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);
        int num = op->numInputs();
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
        checkCnnlError(cnnlCreateTensorDescriptor(&desc));
        checkCnnlError(cnnlSetTensorDescriptor(desc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));
        cnnlTensorDescriptor_t descArray[num];
        for (int i = 0; i < num; ++i) {
            checkCnnlError(cnnlCreateTensorDescriptor(&descArray[i]));
            checkCnnlError(
                cnnlSetTensorDescriptor(descArray[i], CNNL_LAYOUT_NCHW,
                                        CNNL_DTYPE_FLOAT, 4, dim_array));
        }

        cnnlStatus_t stat =
            cnnlMulN(context->cnnlHandle(), descArray, argv, num, desc, cData);
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

REGISTER_KERNEL(Device::BANG, OpType::MulN, DataType::Float32, MulNCnnl,
                "MulN_cnnl_BANG_Float32");
}; // namespace infini

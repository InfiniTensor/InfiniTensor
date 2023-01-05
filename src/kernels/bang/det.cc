#include "operators/det.h"
#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"

namespace infini {
class DetCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<DetObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        DetObj::Mode mode = op->getMode();
        cnnlDetMode_t nlMode;
        if (mode == DetObj::LogDet) {
            nlMode = CNNL_DET_MODE_LOGDET;
        } else {
            nlMode = CNNL_DET_MODE_DET;
        }
        cnnlTensorDescriptor_t aDesc, cDesc;
        auto dimin = op->getInputs(0)->getDims();
        auto dimout = op->getOutput()->getDims();
        if (dimin.size() != 4 || dimout.size() != 2)
            IT_TODO_HALT();

        int dimin_array[4] = {dimin[0], dimin[1], dimin[2], dimin[3]};
        int dimout_array[2] = {dimout[0], dimout[1]};
        // get inputs
        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            aDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 4, dimin_array));

        // get outputs
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            cDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 2, dimout_array));

        cnnlStatus_t stat =
            cnnlDet(context->cnnlHandle(), nlMode, aDesc, aData, cDesc, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        // Destories in BANG does not require sync. But cnnl does not state
        // whether sync is required before destories.
        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::Det, DataType::Float32, DetCnnl,
                "Det_cnnl_BANG_Float32");
}; // namespace infini

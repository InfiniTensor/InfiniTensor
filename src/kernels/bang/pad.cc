#include "operators/pad.h"
#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"

namespace infini {
class PadCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<PadObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        cnnlTensorDescriptor_t aDesc, cDesc;
        auto dimIn = op->getInputs(0)->getDims();
        auto dimOut = op->getOutput()->getDims();

        int dim_size = dimIn.size();
        int paddings[dim_size * 2];

        std::vector<int> pads = op->getPads();

        if (pads.size() == 2 && dim_size != 1) {
            for (int i = 0; i < dim_size * 2; i += 2) {
                paddings[i] = pads[0];
                paddings[i + 1] = pads[1];
            }
        } else {
            for (int i = 0; i < dim_size * 2; i += 2) {
                paddings[i] = pads[i / 2];
                paddings[i + 1] = pads[i / 2 + dim_size];
            }
        }

        float paddingValue = 0.0;
        // input
        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_ARRAY,
                                               CNNL_DTYPE_FLOAT, dimIn.size(),
                                               dimIn.data()));
        // output
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_ARRAY,
                                               CNNL_DTYPE_FLOAT, dimOut.size(),
                                               dimOut.data()));

        cnnlStatus_t stat = cnnlPad(context->cnnlHandle(), aDesc, aData,
                                    paddings, &paddingValue, cDesc, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        // Destories in BANG does not require sync. But cnnl does not state
        // whether sync is required before destories.
        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::Pad, DataType::Float32, PadCnnl,
                "Pad_cnnl_BANG_Float32");

}; // namespace infini

#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"
#include "operators/unary.h"

namespace infini {
class ArangeCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ArangeObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        float start = op->getStartValue();
        float step = op->getStepValue();
        int length = op->getLength();

        cnnlTensorDescriptor_t cDesc;
        int dim_array[1] = {length};
        // get inputs
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_ARRAY,
                                               CNNL_DTYPE_FLOAT, 1, dim_array));

        cnnlStatus_t stat = cnnlArange_v2(context->cnnlHandle(),
                                          CNNL_COMPUTATION_HIGH_PRECISION,
                                          &start, &step, cDesc, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        // Destories in BANG does not require sync. But cnnl does not state
        // whether sync is required before destories.
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::Arange, DataType::Float32, ArangeCnnl,
                "Arange_cnnl_BANG_Float32");

}; // namespace infini

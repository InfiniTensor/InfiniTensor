#include "operators/slice.h"
#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"

namespace infini {
class SliceCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<SliceObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        auto starts = op->getStarts();
        auto ends = op->getEnds();
        auto steps = op->getSteps();

        int32_t starts_array[starts.size()];
        int32_t ends_array[ends.size()];
        int32_t steps_array[steps.size()];

        for (size_t i = 0; i < starts.size(); i++) {
            starts_array[i] = starts[i];
            ends_array[i] = ends[i];
            steps_array[i] = steps[i];
        }

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto aDim = op->getInputs(0)->getDims();
        int aDim_size = aDim.size();
        int aDim_array[aDim_size];
        for (int i = 0; i < aDim_size; ++i) {
            aDim_array[i] = aDim[i];
        }
        auto cDim = op->getOutput()->getDims();
        int cDim_size = cDim.size();
        int cDim_array[cDim_size];
        for (int i = 0; i < cDim_size; ++i) {
            cDim_array[i] = cDim[i];
        }
        cnnlTensorDescriptor_t aDesc, cDesc;
        // input
        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            aDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, aDim_size, aDim_array));
        // output
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            cDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, cDim_size, cDim_array));

        cnnlStatus_t stat =
            cnnlStridedSlice(context->cnnlHandle(), aDesc, aData, starts_array,
                             ends_array, steps_array, cDesc, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::Slice, DataType::Float32, SliceCnnl,
                "Slice_cnnl_BANG_Float32");
}; // namespace infini

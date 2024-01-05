#include "operators/slice.h"
#include "kunlun/kunlun_kernel_without_config.h"
#include "kunlun/kunlun_runtime.h"

namespace infini {
class SliceXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<SliceObj>(_op);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *inData = op->getInputs(0)->getRawDataPtr<void *>();
        void *outData = op->getOutput()->getRawDataPtr<void *>();

        // Get attributes of Slice OP
        Shape starts = op->getStarts(), ends = op->getEnds(),
              steps = op->getSteps();
        Shape inShape = op->getInputs(0)->getDims();
        // If all steps are 1, set continuous True
        bool continuous =
            (size_t)std::count(steps.begin(), steps.end(), 1) == steps.size();
        if (continuous) {
            // if continuous, call xdnn::slice
            checkKUNLUNError(
                xdnn::slice<float>(context->KUNLUNHandle(), (float *)inData,
                                   (float *)outData, inShape, starts, ends));

        } else {
            // else call xdnn::strided_slice
            checkKUNLUNError(xdnn::strided_slice<float>(
                context->KUNLUNHandle(), (float *)inData, (float *)outData,
                inShape, starts, ends, steps));
        }
    }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::Slice, DataType::Float32, SliceXdnn,
                "Slice_xdnn_KUNLUN_Float32")
}; // namespace infini
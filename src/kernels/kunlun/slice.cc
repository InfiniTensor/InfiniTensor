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
            auto ret = 0;
            if (op->getDType() == DataType::Float32) {
                ret =
                    xdnn::slice<float>(context->KUNLUNHandle(), (float *)inData,
                                       (float *)outData, inShape, starts, ends);
            } else if (op->getDType() == DataType::Float16) {
                ret = xdnn::slice<float16>(
                    context->KUNLUNHandle(), (float16 *)inData,
                    (float16 *)outData, inShape, starts, ends);
            } else if (op->getDType() == DataType::Int8) {
                ret = xdnn::slice<int8_t>(context->KUNLUNHandle(),
                                          (int8_t *)inData, (int8_t *)outData,
                                          inShape, starts, ends);
            } else if (op->getDType() == DataType::Int32) {
                ret = xdnn::slice<int>(context->KUNLUNHandle(), (int *)inData,
                                       (int *)outData, inShape, starts, ends);
            } else if (op->getDType() == DataType::Int64) {
                ret = xdnn::slice<int64_t>(
                    context->KUNLUNHandle(), (int64_t *)inData,
                    (int64_t *)outData, inShape, starts, ends);
            } else if (op->getDType() == DataType::Int16) {
                ret = xdnn::slice<int16_t>(
                    context->KUNLUNHandle(), (int16_t *)inData,
                    (int16_t *)outData, inShape, starts, ends);
            } else {
                IT_ASSERT(false, "Unsupported data type");
            }

            assert(ret == 0);

        } else {
            // else call xdnn::strided_slice
            auto ret = 0;
            if (op->getDType() == DataType::Float32) {
                ret = xdnn::strided_slice<float>(
                    context->KUNLUNHandle(), (float *)inData, (float *)outData,
                    inShape, starts, ends, steps);
            } else if (op->getDType() == DataType::Float16) {
                ret = xdnn::strided_slice<float16>(
                    context->KUNLUNHandle(), (float16 *)inData,
                    (float16 *)outData, inShape, starts, ends, steps);
            } else if (op->getDType() == DataType::Int8) {
                ret = xdnn::strided_slice<int8_t>(
                    context->KUNLUNHandle(), (int8_t *)inData,
                    (int8_t *)outData, inShape, starts, ends, steps);
            } else if (op->getDType() == DataType::Int32) {
                ret = xdnn::strided_slice<int>(context->KUNLUNHandle(),
                                               (int *)inData, (int *)outData,
                                               inShape, starts, ends, steps);
            } else if (op->getDType() == DataType::Int64) {
                ret = xdnn::strided_slice<int64_t>(
                    context->KUNLUNHandle(), (int64_t *)inData,
                    (int64_t *)outData, inShape, starts, ends, steps);
            } else if (op->getDType() == DataType::Int16) {
                ret = xdnn::strided_slice<int16_t>(
                    context->KUNLUNHandle(), (int16_t *)inData,
                    (int16_t *)outData, inShape, starts, ends, steps);
            } else {
                IT_ASSERT(false, "Unsupported data type");
            }

            assert(ret == 0);
        }
    }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::Slice, SliceXdnn, "Slice_xdnn_KUNLUN")
}; // namespace infini


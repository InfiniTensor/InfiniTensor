#include "operators/reshape.h"
#include "kunlun/kunlun_kernel_without_config.h"
#include "kunlun/kunlun_runtime.h"

namespace infini {
class CopyXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &op,
                 const RuntimeObj *_context) const override {
        auto inData = op->getInputs(0)->getRawDataPtr<void *>();
        auto outData = op->getOutputs()[0]->getRawDataPtr<void *>();
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);
        auto dtype = op->getDType();
        auto len = op->getInputs(0)->size();
        if (dtype == DataType::Float32) {
            checkKUNLUNError(
                (xdnn::copy<float>(context->KUNLUNHandle(), (float *)inData,
                                   (float *)outData, len)));
        } else if (dtype == DataType::Float16) {
            checkKUNLUNError(
                (xdnn::copy<float16>(context->KUNLUNHandle(), (float16 *)inData,
                                     (float16 *)outData, len)));
        } else if (dtype == DataType::Int8) {
            checkKUNLUNError(
                (xdnn::copy<int8_t>(context->KUNLUNHandle(), (int8_t *)inData,
                                    (int8_t *)outData, len)));
        } else if (dtype == DataType::Int32) {
            checkKUNLUNError(
                (xdnn::copy<int8_t>(context->KUNLUNHandle(), (int8_t *)inData,
                                    (int8_t *)outData, len)));

        } else if (dtype == DataType::Int64) {
            checkKUNLUNError(
                (xdnn::copy<int8_t>(context->KUNLUNHandle(), (int8_t *)inData,
                                    (int8_t *)outData, len)));

        } else if (dtype == DataType::Int16) {
            checkKUNLUNError(
                (xdnn::copy<int8_t>(context->KUNLUNHandle(), (int8_t *)inData,
                                    (int8_t *)outData, len)));
        } else {
            IT_ASSERT(false,
                      "unsupported data type " + op->getDType().toString());
        }
    }
};

// reshape/flatten/identity all act as copying from input to output.
REGISTER_KERNEL(Device::KUNLUN, OpType::Reshape, CopyXdnn, "Reshape_Xdnn");
REGISTER_KERNEL(Device::KUNLUN, OpType::Flatten, CopyXdnn, "Flatten_Xdnn");
REGISTER_KERNEL(Device::KUNLUN, OpType::Identity, CopyXdnn, "Identity_Xdnn");
REGISTER_KERNEL(Device::KUNLUN, OpType::Squeeze, CopyXdnn, "Squeeze_Xdnn");
REGISTER_KERNEL(Device::KUNLUN, OpType::Unsqueeze, CopyXdnn, "Unsqueeze_Xdnn");

}; // namespace infini

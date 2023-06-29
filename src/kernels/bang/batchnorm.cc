#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"
#include "operators/batch_norm.h"

namespace infini {
class BatchNormCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<BatchNormObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const input = (op->getInputs(0)->getRawDataPtr<void *>());
        // void *const mean = (op->getInputs(1)->getRawDataPtr<void *>());
        // void *const var = (op->getInputs(2)->getRawDataPtr<void *>());
        // void *const scale = (op->getInputs(3)->getRawDataPtr<void *>());
        // void *const bias = (op->getInputs(4)->getRawDataPtr<void *>());
        void *const output = (op->getOutput()->getRawDataPtr<void *>());

        auto dims = op->getInputs(0)->getDims();

        if (dims.size() != 4)
            IT_TODO_HALT();

        // get inputs
        cnnlTensorDescriptor_t inDesc;
        checkCnnlError(cnnlCreateTensorDescriptor(&inDesc));
        checkCnnlError(cnnlSetTensorDescriptor(inDesc, CNNL_LAYOUT_NHWC,
                                                 CNNL_DTYPE_FLOAT, dims.size(),
                                                 dims.data()));

        cnnlStatus_t stat = cnnlCopy(context->cnnlHandle(), inDesc, input, inDesc, output);

        if (stat != CNNL_STATUS_SUCCESS)
            return;

        checkCnnlError(cnnlDestroyTensorDescriptor(inDesc));
    }
};

class BatchNormNHWCCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<BatchNormNHWCObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const input = (op->getInputs(0)->getRawDataPtr<void *>());
        // void *const mean = (op->getInputs(1)->getRawDataPtr<void *>());
        // void *const var = (op->getInputs(2)->getRawDataPtr<void *>());
        // void *const scale = (op->getInputs(3)->getRawDataPtr<void *>());
        // void *const bias = (op->getInputs(4)->getRawDataPtr<void *>());
        void *const output = (op->getOutput()->getRawDataPtr<void *>());

        auto dims = op->getInputs(0)->getDims();

        if (dims.size() != 4)
            IT_TODO_HALT();

        // get inputs
        cnnlTensorDescriptor_t inDesc;
        checkCnnlError(cnnlCreateTensorDescriptor(&inDesc));
        checkCnnlError(cnnlSetTensorDescriptor(inDesc, CNNL_LAYOUT_NHWC,
                                                 CNNL_DTYPE_FLOAT, dims.size(),
                                                 dims.data()));

        cnnlStatus_t stat = cnnlCopy(context->cnnlHandle(), inDesc, input, inDesc, output);

        if (stat != CNNL_STATUS_SUCCESS)
            return;

        checkCnnlError(cnnlDestroyTensorDescriptor(inDesc));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::BatchNorm, DataType::Float32,
                BatchNormCnnl, "BatchNorm_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::BatchNormNHWC, DataType::Float32,
                BatchNormNHWCCnnl, "BatchNormNHWC_cnnl_BANG_Float32");

}; // namespace infini

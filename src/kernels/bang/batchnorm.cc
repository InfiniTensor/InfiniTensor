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
        void *const mean = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const var = (op->getInputs(2)->getRawDataPtr<void *>());
        void *const scale = (op->getInputs(3)->getRawDataPtr<void *>());
        void *const bias = (op->getInputs(4)->getRawDataPtr<void *>());
        void *const output = (op->getOutput()->getRawDataPtr<void *>());

        auto dims = op->getInputs(0)->getDims();

        if (dims.size() != 4)
            IT_TODO_HALT();

        int dimArray[4], strideArray[4], dimPArray[1], stridePArray[1];

        for (size_t i = 0; i < dims.size(); ++i) {
            dimArray[i] = dims[i];
            strideArray[i] = op->getInputs(0)->getStride()[i];
        }
        int w = dimArray[3];
        dimArray[3] = dimArray[1];
        int h = dimArray[2];
        dimArray[1] = h;
        dimArray[2] = w;

        dimPArray[0] = op->getInputs(1)->getDims()[0];
        stridePArray[0] = op->getInputs(1)->getDims()[0];
        // get inputs
        cnnlTensorDescriptor_t inDesc;
        checkCnnlError(cnnlCreateTensorDescriptor(&inDesc));
        checkCnnlError(cnnlSetTensorDescriptorEx(inDesc, CNNL_LAYOUT_NHWC,
                                                 CNNL_DTYPE_FLOAT, dims.size(),
                                                 dimArray, strideArray));

        // get bnScaleBiasMeanVarDesc
        cnnlTensorDescriptor_t paraDesc;
        checkCnnlError(cnnlCreateTensorDescriptor(&paraDesc));
        checkCnnlError(cnnlSetTensorDescriptorEx(paraDesc, CNNL_LAYOUT_ARRAY,
                                                 CNNL_DTYPE_FLOAT, 1, dimPArray,
                                                 stridePArray));

        float alpha = 1.f, beta = 0.f;
        // This mode is intended for use after convolutional layers
        cnnlStatus_t stat = cnnlBatchNormForwardInference(
            context->cnnlHandle(), &alpha, &beta, inDesc, input, paraDesc,
            scale, bias, mean, var, op->getEps(), inDesc, output);

        if (stat != CNNL_STATUS_SUCCESS)
            return;

        // Destories in BANG does not require sync. But cnnl does not state
        // whether sync is required before destories.
        checkCnnlError(cnnlDestroyTensorDescriptor(inDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(paraDesc));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::BatchNormalization, DataType::Float32,
                BatchNormCnnl, "BatchNorm_cnnl_BANG_Float32");

}; // namespace infini

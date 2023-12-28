#include "operators/dropout.h"
#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"

namespace infini {
class DropoutCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<DropoutObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const iData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const oData = (op->getOutput(0)->getRawDataPtr<void *>());
        void *const mData = (op->getOutput(1)->getRawDataPtr<void *>());

        cnnlRandGenerator_t generator;
        cnnlRandCreateGenerator(&generator, CNNL_RAND_RNG_FAST);
        cnnlRandSetPseudoRandomGeneratorSeed(generator, 233);
        cnnlRandSetMTGP32Period(generator, CNNL_RAND_MTGP32_P11213);

        cnnlTensorDescriptor_t oDesc;
        auto oDim = op->getOutput(0)->getDims();
        checkCnnlError(cnnlCreateTensorDescriptor(&oDesc));
        checkCnnlError(cnnlSetTensorDescriptor(oDesc, CNNL_LAYOUT_ARRAY,
                                               CNNL_DTYPE_FLOAT, oDim.size(),
                                               oDim.data()));

        auto ratio = op->getRatio();
        // auto train = op->getTrainingMode();

        cnnlStatus_t stat =
            cnnlFusedDropout_v2(context->cnnlHandle(), generator, oDesc, iData,
                                ratio, NULL, oDesc, mData, oDesc, oData);

        if (stat != CNNL_STATUS_SUCCESS)
            return;

        checkCnnlError(cnnlDestroyTensorDescriptor(oDesc));
        checkCnnlError(cnnlRandDestroyGenerator(generator));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::Dropout, DataType::Float32, DropoutCnnl,
                "Dropout_cnnl_BANG_Float32");

}; // namespace infini

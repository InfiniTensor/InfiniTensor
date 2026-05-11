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

        if (op->getInputs(0)->getRank() == 3 && dimin.at(2) == 1) {
            std::swap(dimin[0], dimin[2]);
        }

        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            aDesc, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(op->getDType()),
            dimin.size(), dimin.data()));

        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            cDesc, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(op->getDType()),
            dimout.size(), dimout.data()));

        cnnlStatus_t stat =
            cnnlDet(context->cnnlHandle(), nlMode, aDesc, aData, cDesc, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::Det, DetCnnl, "Det_cnnl_BANG");
}; // namespace infini

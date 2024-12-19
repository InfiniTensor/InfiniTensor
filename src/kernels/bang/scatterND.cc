#include "operators/scatterND.h"
#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"

namespace infini {
class ScatterNDCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ScatterNDObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getInputs(2)->getRawDataPtr<void *>());
        void *const dData = (op->getOutput()->getRawDataPtr<void *>());

        cnnlTensorDescriptor_t aDesc, bDesc, cDesc, dDesc;
        auto aDim = op->getInputs(0)->getDims();
        auto bDim = op->getInputs(1)->getDims();
        auto cDim = op->getInputs(2)->getDims();
        auto dDim = op->getOutput()->getDims();

        if (aDim.size() == 0) {
            aDim.push_back(1);
        }
        if (bDim.size() == 0) {
            bDim.push_back(1);
        }
        if (cDim.size() == 0) {
            cDim.push_back(1);
        }
        if (dDim.size() == 0) {
            dDim.push_back(1);
        }

        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            aDesc, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(op->getDType()),
            aDim.size(), aDim.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&bDesc));
        checkCnnlError(cnnlSetTensorDescriptor(bDesc, CNNL_LAYOUT_ARRAY,
                                               CNNL_DTYPE_INT64, bDim.size(),
                                               bDim.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            cDesc, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(op->getDType()),
            cDim.size(), cDim.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&dDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            dDesc, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(op->getDType()),
            dDim.size(), dDim.data()));
        cnnlScatterNdMode_t mode = CNNL_SCATTERND_UPDATE;
        cnnlStatus_t stat =
            cnnlScatterNd_v2(context->cnnlHandle(), mode, bDesc, bData, cDesc,
                             cData, aDesc, aData, dDesc, dData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(bDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(dDesc));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::ScatterND, ScatterNDCnnl,
                "ScatterND_cnnl_BANG");

}; // namespace infini

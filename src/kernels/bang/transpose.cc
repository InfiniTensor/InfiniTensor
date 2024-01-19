#include "operators/transpose.h"
#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"

namespace infini {
class TransposeCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<TransposeObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        cnnlTensorDescriptor_t aDesc, cDesc;
        auto dimin = op->getInputs(0)->getDims();
        auto dimout = op->getOutput()->getDims();

        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            aDesc, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(op->getDType()),
            dimin.size(), dimin.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            cDesc, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(op->getDType()),
            dimout.size(), dimout.data()));

        auto permute = op->getPermute();
        cnnlTransposeDescriptor_t opDesc;
        checkCnnlError(cnnlCreateTransposeDescriptor(&opDesc));
        checkCnnlError(
            cnnlSetTransposeDescriptor(opDesc, permute.size(), permute.data()));

        size_t wsSize;
        cnnlGetTransposeWorkspaceSize(context->cnnlHandle(), aDesc, opDesc,
                                      &wsSize);
        BangPtr wsData = context->getWorkspace(wsSize);

        cnnlStatus_t stat =
            cnnlTranspose_v2(context->cnnlHandle(), opDesc, aDesc, aData, cDesc,
                             cData, wsData, wsSize);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
        checkCnnlError(cnnlDestroyTransposeDescriptor(opDesc));
    }
};

class DepthToSpaceCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<DepthToSpaceObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto reshape = op->getReshapeDim();
        auto transpose = op->getTransposeDim();
        auto mode = op->getMode();

        std::vector<int> permute;
        if (mode == 0) {
            permute = {0, 3, 4, 1, 5, 2};
        } else {
            permute = {0, 1, 4, 2, 5, 3};
        }

        cnnlTensorDescriptor_t aDesc, cDesc;
        auto dimout = op->getOutput()->getDims();

        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            aDesc, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(op->getDType()),
            reshape.size(), reshape.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            cDesc, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(op->getDType()),
            transpose.size(), transpose.data()));

        cnnlTransposeDescriptor_t opDesc;
        checkCnnlError(cnnlCreateTransposeDescriptor(&opDesc));
        checkCnnlError(
            cnnlSetTransposeDescriptor(opDesc, permute.size(), permute.data()));

        size_t wsSize;
        cnnlGetTransposeWorkspaceSize(context->cnnlHandle(), aDesc, opDesc,
                                      &wsSize);
        BangPtr wsData = context->getWorkspace(wsSize);

        cnnlStatus_t stat =
            cnnlTranspose_v2(context->cnnlHandle(), opDesc, aDesc, aData, cDesc,
                             cData, wsData, wsSize);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
        checkCnnlError(cnnlDestroyTransposeDescriptor(opDesc));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::Transpose, TransposeCnnl,
                "Transpose_cnnl_BANG");

REGISTER_KERNEL(Device::BANG, OpType::DepthToSpace, DepthToSpaceCnnl,
                "DepthToSpace_cnnl_BANG");
}; // namespace infini

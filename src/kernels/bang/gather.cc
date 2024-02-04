#include "operators/gather.h"
#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"

namespace infini {
class GatherCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<GatherObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        cnnlTensorDescriptor_t aDesc, bDesc, cDesc;
        auto aDim = op->getInputs(0)->getDims();
        auto bDim = op->getInputs(1)->getDims();
        auto cDim = op->getOutput()->getDims();

        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            aDesc, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(op->getDType()),
            aDim.size(), aDim.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&bDesc));

        if (bDim.size() == 0) {
            bDim.push_back(1);
        }
        checkCnnlError(cnnlSetTensorDescriptor(bDesc, CNNL_LAYOUT_ARRAY,
                                               CNNL_DTYPE_INT32, bDim.size(),
                                               bDim.data()));

        BangPtr indices;
        DataType indicesDataType = op->getInputs(1)->getDType();
        if (indicesDataType == DataType::Int64) {
            // cnnlGatherV2 does not support int64 indices
            int indicesSize =
                op->getInputs(1)->getBytes() / indicesDataType.getSize();
            indices = context->getWorkspace(indicesSize * sizeof(int));
            cnnlTensorDescriptor_t bDescInt64;
            checkCnnlError(cnnlCreateTensorDescriptor(&bDescInt64));
            checkCnnlError(cnnlSetTensorDescriptor(
                bDescInt64, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT64, bDim.size(),
                bDim.data()));
            checkCnnlError(cnnlCastDataType(context->cnnlHandle(), bDescInt64,
                                            bData, CNNL_CAST_INT64_TO_INT32,
                                            bDesc, indices));
            cnrtQueueSync(context->getBangQueue());
            checkCnnlError(cnnlDestroyTensorDescriptor(bDescInt64));
        } else if (indicesDataType == DataType::Int32) {
            indices = bData;
        } else {
            IT_TODO_HALT_MSG("Unsupported data type of indices: " +
                             indicesDataType.toString());
        }

        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            cDesc, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(op->getDType()),
            cDim.size(), cDim.data()));

        BangPtr wsData = context->getWorkspace(aDim.size() * sizeof(int));
        context->copyBlobFromCPU(wsData, aDim.data(),
                                 aDim.size() * sizeof(int));

        auto axis = op->getAxis();
        cnnlStatus_t stat =
            cnnlGatherV2(context->cnnlHandle(), axis, aDesc, aData,
                         reinterpret_cast<const int *>(wsData), bDesc,
                         reinterpret_cast<const int *>(indices), cDesc, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(bDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::Gather, GatherCnnl, "Gather_cnnl_BANG");

}; // namespace infini

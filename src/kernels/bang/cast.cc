#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"
#include "operators/unary.h"

namespace infini {
class CastCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<CastObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        cnnlTensorDescriptor_t aDesc, cDesc;
        auto dim = op->getInputs(0)->getDims();
        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        cnnlCastDataType_t NlCastType;
        CastObj::CastType type = op->getType();
        switch (type) {
        case CastObj::Float2Int64:
            checkCnnlError(cnnlSetTensorDescriptor(
                aDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_FLOAT, dim.size(), dim.data()));
            checkCnnlError(cnnlSetTensorDescriptor(
                cDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_INT64, dim.size(), dim.data()));
            NlCastType = CNNL_CAST_FLOAT_TO_INT64;
            break;
        case CastObj::Float2Int32:
            checkCnnlError(cnnlSetTensorDescriptor(
                aDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_FLOAT, dim.size(), dim.data()));
            checkCnnlError(cnnlSetTensorDescriptor(
                cDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_INT32, dim.size(), dim.data()));
            NlCastType = CNNL_CAST_FLOAT_TO_INT32;
            break;
        case CastObj::Float2Int16:
            checkCnnlError(cnnlSetTensorDescriptor(
                aDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_FLOAT, dim.size(), dim.data()));
            checkCnnlError(cnnlSetTensorDescriptor(
                cDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_INT16, dim.size(), dim.data()));
            NlCastType = CNNL_CAST_FLOAT_TO_INT16;
            break;
        case CastObj::Float2Int8:
            checkCnnlError(cnnlSetTensorDescriptor(
                aDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_FLOAT, dim.size(), dim.data()));
            checkCnnlError(cnnlSetTensorDescriptor(
                cDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_INT8, dim.size(), dim.data()));
            NlCastType = CNNL_CAST_FLOAT_TO_INT8;
            break;
        case CastObj::Int322Float:
            checkCnnlError(cnnlSetTensorDescriptor(
                aDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_INT32, dim.size(), dim.data()));
            checkCnnlError(cnnlSetTensorDescriptor(
                cDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_FLOAT, dim.size(), dim.data()));
            NlCastType = CNNL_CAST_INT32_TO_FLOAT;
            break;
        case CastObj::Int322Int8:
            checkCnnlError(cnnlSetTensorDescriptor(
                aDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_INT32, dim.size(), dim.data()));
            checkCnnlError(cnnlSetTensorDescriptor(
                cDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_INT8, dim.size(), dim.data()));
            NlCastType = CNNL_CAST_INT32_TO_INT8;
            break;
        case CastObj::Int322Int16:
            checkCnnlError(cnnlSetTensorDescriptor(
                aDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_INT32, dim.size(), dim.data()));
            checkCnnlError(cnnlSetTensorDescriptor(
                cDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_INT16, dim.size(), dim.data()));
            NlCastType = CNNL_CAST_INT32_TO_INT16;
            break;
        case CastObj::Int162Float:
            checkCnnlError(cnnlSetTensorDescriptor(
                aDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_INT16, dim.size(), dim.data()));
            checkCnnlError(cnnlSetTensorDescriptor(
                cDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_FLOAT, dim.size(), dim.data()));
            NlCastType = CNNL_CAST_INT16_TO_FLOAT;
            break;
        case CastObj::Int162Int32:
            checkCnnlError(cnnlSetTensorDescriptor(
                aDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_INT16, dim.size(), dim.data()));
            checkCnnlError(cnnlSetTensorDescriptor(
                cDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_INT32, dim.size(), dim.data()));
            NlCastType = CNNL_CAST_INT16_TO_INT32;
            break;
        case CastObj::Int82Float:
            checkCnnlError(cnnlSetTensorDescriptor(
                aDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_INT8, dim.size(), dim.data()));
            checkCnnlError(cnnlSetTensorDescriptor(
                cDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_FLOAT, dim.size(), dim.data()));
            NlCastType = CNNL_CAST_INT8_TO_FLOAT;
            break;
        case CastObj::Int82Int16:
            checkCnnlError(cnnlSetTensorDescriptor(
                aDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_INT8, dim.size(), dim.data()));
            checkCnnlError(cnnlSetTensorDescriptor(
                cDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_INT16, dim.size(), dim.data()));
            NlCastType = CNNL_CAST_INT8_TO_INT16;
            break;
        case CastObj::Int82Int32:
            checkCnnlError(cnnlSetTensorDescriptor(
                aDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_INT8, dim.size(), dim.data()));
            checkCnnlError(cnnlSetTensorDescriptor(
                cDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_INT32, dim.size(), dim.data()));
            NlCastType = CNNL_CAST_INT8_TO_INT32;
            break;
        case CastObj::Uint82Float:
            checkCnnlError(cnnlSetTensorDescriptor(
                aDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_UINT8, dim.size(), dim.data()));
            checkCnnlError(cnnlSetTensorDescriptor(
                cDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_FLOAT, dim.size(), dim.data()));
            NlCastType = CNNL_CAST_UINT8_TO_FLOAT;
            break;
        case CastObj::Uint82Int32:
            checkCnnlError(cnnlSetTensorDescriptor(
                aDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_UINT8, dim.size(), dim.data()));
            checkCnnlError(cnnlSetTensorDescriptor(
                cDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_INT32, dim.size(), dim.data()));
            NlCastType = CNNL_CAST_UINT8_TO_INT32;
            break;
        case CastObj::Uint82Int64:
            checkCnnlError(cnnlSetTensorDescriptor(
                aDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_UINT8, dim.size(), dim.data()));
            checkCnnlError(cnnlSetTensorDescriptor(
                cDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_INT64, dim.size(), dim.data()));
            NlCastType = CNNL_CAST_UINT8_TO_INT64;
            break;
        case CastObj::Int322Int64:
            checkCnnlError(cnnlSetTensorDescriptor(
                aDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_INT32, dim.size(), dim.data()));
            checkCnnlError(cnnlSetTensorDescriptor(
                cDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_INT64, dim.size(), dim.data()));
            NlCastType = CNNL_CAST_INT32_TO_INT64;
            break;
        case CastObj::Int642Int32:
            checkCnnlError(cnnlSetTensorDescriptor(
                aDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_INT64, dim.size(), dim.data()));
            checkCnnlError(cnnlSetTensorDescriptor(
                cDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_INT32, dim.size(), dim.data()));
            NlCastType = CNNL_CAST_INT64_TO_INT32;
            break;
        case CastObj::Int642Uint32:
            checkCnnlError(cnnlSetTensorDescriptor(
                aDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_INT64, dim.size(), dim.data()));
            checkCnnlError(cnnlSetTensorDescriptor(
                cDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_UINT32, dim.size(), dim.data()));
            NlCastType = CNNL_CAST_INT64_TO_UINT32;
            break;
        case CastObj::Int642Float:
            checkCnnlError(cnnlSetTensorDescriptor(
                aDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_INT64, dim.size(), dim.data()));
            checkCnnlError(cnnlSetTensorDescriptor(
                cDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_FLOAT, dim.size(), dim.data()));
            NlCastType = CNNL_CAST_INT64_TO_FLOAT;
            break;
        case CastObj::Uint322Int64:
            checkCnnlError(cnnlSetTensorDescriptor(
                aDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_UINT32, dim.size(), dim.data()));
            checkCnnlError(cnnlSetTensorDescriptor(
                cDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_INT64, dim.size(), dim.data()));
            NlCastType = CNNL_CAST_UINT32_TO_INT64;
            break;
        default:
            IT_TODO_HALT();
        }
        cnnlStatus_t stat = cnnlCastDataType(context->cnnlHandle(), aDesc,
                                             aData, NlCastType, cDesc, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::Cast, DataType::Float32, CastCnnl,
                "Cast_cnnl_BANG_Float32");

}; // namespace infini

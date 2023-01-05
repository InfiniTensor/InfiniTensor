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
        if (dim.size() != 4)
            IT_TODO_HALT();

        int dim_array[4] = {dim[0], dim[1], dim[2], dim[3]};
        // get inputs
        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        cnnlCastDataType_t NlCastType;
        CastObj::CastType type = op->getType();
        switch (type) {
        case CastObj::Float2Half:
            checkCnnlError(cnnlSetTensorDescriptor(
                aDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_FLOAT, 4, dim_array));
            checkCnnlError(cnnlSetTensorDescriptor(
                cDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_HALF, 4, dim_array));
            NlCastType = CNNL_CAST_FLOAT_TO_HALF;
            break;
        case CastObj::Float2HalfIEEE754:
        case CastObj::Float2Double:
        case CastObj::Float2Int64:
        case CastObj::Float2Int32:
            checkCnnlError(cnnlSetTensorDescriptor(
                aDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_FLOAT, 4, dim_array));
            checkCnnlError(cnnlSetTensorDescriptor(
                cDesc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_INT32, 4, dim_array));
            NlCastType = CNNL_CAST_FLOAT_TO_INT32;
        case CastObj::Float2Int16:
        case CastObj::Float2Int8:
        case CastObj::Float2Bool:
            // Todo
            break;
        case CastObj::Half2Float:
        case CastObj::Half2Int32:
        case CastObj::Half2Int64:
        case CastObj::Half2Int16:
        case CastObj::Half2Int8:
        case CastObj::Half2Uint8:
        case CastObj::Half2Bool:
        case CastObj::Half2FloatInf:
            // todo
            break;
        case CastObj::Int322Float:
        case CastObj::Int322Half:
        case CastObj::Int322Int8:
        case CastObj::Int322Int16:
            // todo
            break;
        case CastObj::Int162Float:
        case CastObj::Int162Half:
        case CastObj::Int162Int32:
            // todo
            break;
        case CastObj::Int82Float:
        case CastObj::Int82Half:
        case CastObj::Int82Int16:
        case CastObj::Int82Int32:
            // todo
            break;
        case CastObj::Uint82Float:
        case CastObj::Uint82Half:
        case CastObj::Uint82Int32:
        case CastObj::Uint82Int64:
            // todo
            break;
        case CastObj::Bool2Float:
        case CastObj::Bool2Half:
        case CastObj::Bool2Int32:
            // todo
            break;
        case CastObj::Int322Int64:
        case CastObj::Int322Bool:
            // todo
            break;
        case CastObj::Int642Int32:
        case CastObj::Int642Uint32:
        case CastObj::Int642Float:
        case CastObj::Int642Half:
            // todo
            break;
        case CastObj::Uint642Uint32:
        case CastObj::Uint322Int64:
        case CastObj::Uint322Uint64:
            // todo
            break;
        case CastObj::Double2Float:
            // todo
            break;
        }
        cnnlStatus_t stat = cnnlCastDataType(context->cnnlHandle(), aDesc,
                                             aData, NlCastType, cDesc, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        // Destories in BANG does not require sync. But cnnl does not state
        // whether sync is required before destories.
        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::Cast, DataType::Float32, CastCnnl,
                "Cast_cnnl_BANG_Float32");

}; // namespace infini

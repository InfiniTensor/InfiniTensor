#include "aclnnop/level2/aclnn_cast.h"
#include "ascend/ascend_kernel_without_config.h"
#include "ascend/ascend_runtime.h"
#include "operators/unary.h"

namespace infini {
class CastAclnn : public ASCENDKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<CastObj>(_op);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto inputD = op->getInputs(0)->getDims();
        auto inputS = op->getInputs(0)->getStride();
        auto outD = op->getOutput()->getDims();
        auto outS = op->getOutput()->getStride();

        std::vector<int64_t> inputDim = castTo64(inputD);
        std::vector<int64_t> inputStride = castTo64(inputS);
        std::vector<int64_t> outputDim = castTo64(outD);
        std::vector<int64_t> outputStride = castTo64(outS);

        aclDataType NlCastType;
        CastType type = op->getType();
        aclTensor *inputTensor = nullptr;
        aclTensor *outputTensor = nullptr;
        switch (type) {
        case CastType::Float2Int64:
            inputTensor = aclCreateTensor(
                inputDim.data(), inputDim.size(), aclDataType::ACL_FLOAT,
                inputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                inputDim.data(), inputDim.size(), aData);
            outputTensor = aclCreateTensor(
                outputDim.data(), outputDim.size(), aclDataType::ACL_INT64,
                outputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                outputDim.data(), outputDim.size(), cData);
            NlCastType = aclDataType::ACL_INT64;
            break;
        case CastType::Float2Int32:
            inputTensor = aclCreateTensor(
                inputDim.data(), inputDim.size(), aclDataType::ACL_FLOAT,
                inputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                inputDim.data(), inputDim.size(), aData);
            outputTensor = aclCreateTensor(
                outputDim.data(), outputDim.size(), aclDataType::ACL_INT32,
                outputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                outputDim.data(), outputDim.size(), cData);
            NlCastType = aclDataType::ACL_INT32;
            break;
        case CastType::Float2Int16:
            inputTensor = aclCreateTensor(
                inputDim.data(), inputDim.size(), aclDataType::ACL_FLOAT,
                inputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                inputDim.data(), inputDim.size(), aData);
            outputTensor = aclCreateTensor(
                outputDim.data(), outputDim.size(), aclDataType::ACL_INT16,
                outputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                outputDim.data(), outputDim.size(), cData);
            NlCastType = aclDataType::ACL_INT16;
            break;
        case CastType::Float2Int8:
            inputTensor = aclCreateTensor(
                inputDim.data(), inputDim.size(), aclDataType::ACL_FLOAT,
                inputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                inputDim.data(), inputDim.size(), aData);
            outputTensor = aclCreateTensor(
                outputDim.data(), outputDim.size(), aclDataType::ACL_INT8,
                outputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                outputDim.data(), outputDim.size(), cData);
            NlCastType = aclDataType::ACL_INT8;
            break;
        case CastType::Int322Float:
            inputTensor = aclCreateTensor(
                inputDim.data(), inputDim.size(), aclDataType::ACL_INT32,
                inputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                inputDim.data(), inputDim.size(), aData);
            outputTensor = aclCreateTensor(
                outputDim.data(), outputDim.size(), aclDataType::ACL_FLOAT,
                outputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                outputDim.data(), outputDim.size(), cData);
            NlCastType = aclDataType::ACL_FLOAT;
            break;
        case CastType::Int322Int8:
            inputTensor = aclCreateTensor(
                inputDim.data(), inputDim.size(), aclDataType::ACL_INT32,
                inputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                inputDim.data(), inputDim.size(), aData);
            outputTensor = aclCreateTensor(
                outputDim.data(), outputDim.size(), aclDataType::ACL_INT8,
                outputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                outputDim.data(), outputDim.size(), cData);
            NlCastType = aclDataType::ACL_INT8;
            break;
        case CastType::Int322Int16:
            inputTensor = aclCreateTensor(
                inputDim.data(), inputDim.size(), aclDataType::ACL_INT32,
                inputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                inputDim.data(), inputDim.size(), aData);
            outputTensor = aclCreateTensor(
                outputDim.data(), outputDim.size(), aclDataType::ACL_INT16,
                outputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                outputDim.data(), outputDim.size(), cData);
            NlCastType = aclDataType::ACL_INT16;
            break;
        case CastType::Int162Float:
            inputTensor = aclCreateTensor(
                inputDim.data(), inputDim.size(), aclDataType::ACL_INT16,
                inputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                inputDim.data(), inputDim.size(), aData);
            outputTensor = aclCreateTensor(
                outputDim.data(), outputDim.size(), aclDataType::ACL_FLOAT,
                outputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                outputDim.data(), outputDim.size(), cData);
            NlCastType = aclDataType::ACL_FLOAT;
            break;
        case CastType::Int162Int32:
            inputTensor = aclCreateTensor(
                inputDim.data(), inputDim.size(), aclDataType::ACL_INT16,
                inputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                inputDim.data(), inputDim.size(), aData);
            outputTensor = aclCreateTensor(
                outputDim.data(), outputDim.size(), aclDataType::ACL_INT32,
                outputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                outputDim.data(), outputDim.size(), cData);
            NlCastType = aclDataType::ACL_INT32;
            break;
        case CastType::Int82Float:
            inputTensor = aclCreateTensor(
                inputDim.data(), inputDim.size(), aclDataType::ACL_INT8,
                inputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                inputDim.data(), inputDim.size(), aData);
            outputTensor = aclCreateTensor(
                outputDim.data(), outputDim.size(), aclDataType::ACL_FLOAT,
                outputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                outputDim.data(), outputDim.size(), cData);
            NlCastType = aclDataType::ACL_FLOAT;
            break;
        case CastType::Int82Int16:
            inputTensor = aclCreateTensor(
                inputDim.data(), inputDim.size(), aclDataType::ACL_INT8,
                inputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                inputDim.data(), inputDim.size(), aData);
            outputTensor = aclCreateTensor(
                outputDim.data(), outputDim.size(), aclDataType::ACL_INT16,
                outputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                outputDim.data(), outputDim.size(), cData);
            NlCastType = aclDataType::ACL_INT16;
            break;
        case CastType::Int82Int32:
            inputTensor = aclCreateTensor(
                inputDim.data(), inputDim.size(), aclDataType::ACL_INT8,
                inputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                inputDim.data(), inputDim.size(), aData);
            outputTensor = aclCreateTensor(
                outputDim.data(), outputDim.size(), aclDataType::ACL_INT32,
                outputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                outputDim.data(), outputDim.size(), cData);
            NlCastType = aclDataType::ACL_INT32;
            break;
        case CastType::Uint82Float:
            inputTensor = aclCreateTensor(
                inputDim.data(), inputDim.size(), aclDataType::ACL_UINT8,
                inputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                inputDim.data(), inputDim.size(), aData);
            outputTensor = aclCreateTensor(
                outputDim.data(), outputDim.size(), aclDataType::ACL_FLOAT,
                outputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                outputDim.data(), outputDim.size(), cData);
            NlCastType = aclDataType::ACL_FLOAT;
            break;
        case CastType::Uint82Int32:
            inputTensor = aclCreateTensor(
                inputDim.data(), inputDim.size(), aclDataType::ACL_UINT8,
                inputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                inputDim.data(), inputDim.size(), aData);
            outputTensor = aclCreateTensor(
                outputDim.data(), outputDim.size(), aclDataType::ACL_INT32,
                outputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                outputDim.data(), outputDim.size(), cData);
            NlCastType = aclDataType::ACL_INT32;
            break;
        case CastType::Uint82Int64:
            inputTensor = aclCreateTensor(
                inputDim.data(), inputDim.size(), aclDataType::ACL_UINT8,
                inputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                inputDim.data(), inputDim.size(), aData);
            outputTensor = aclCreateTensor(
                outputDim.data(), outputDim.size(), aclDataType::ACL_INT64,
                outputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                outputDim.data(), outputDim.size(), cData);
            NlCastType = aclDataType::ACL_INT64;
            break;
        case CastType::Int322Int64:
            inputTensor = aclCreateTensor(
                inputDim.data(), inputDim.size(), aclDataType::ACL_INT32,
                inputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                inputDim.data(), inputDim.size(), aData);
            outputTensor = aclCreateTensor(
                outputDim.data(), outputDim.size(), aclDataType::ACL_INT64,
                outputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                outputDim.data(), outputDim.size(), cData);
            NlCastType = aclDataType::ACL_INT64;
            break;
        case CastType::Int642Int32:
            inputTensor = aclCreateTensor(
                inputDim.data(), inputDim.size(), aclDataType::ACL_INT64,
                inputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                inputDim.data(), inputDim.size(), aData);
            outputTensor = aclCreateTensor(
                outputDim.data(), outputDim.size(), aclDataType::ACL_INT32,
                outputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                outputDim.data(), outputDim.size(), cData);
            NlCastType = aclDataType::ACL_INT32;
            break;
        case CastType::Int642Uint32:
            inputTensor = aclCreateTensor(
                inputDim.data(), inputDim.size(), aclDataType::ACL_INT64,
                inputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                inputDim.data(), inputDim.size(), aData);
            outputTensor = aclCreateTensor(
                outputDim.data(), outputDim.size(), aclDataType::ACL_UINT32,
                outputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                outputDim.data(), outputDim.size(), cData);
            NlCastType = aclDataType::ACL_UINT32;
            break;
        case CastType::Int642Float:
            inputTensor = aclCreateTensor(
                inputDim.data(), inputDim.size(), aclDataType::ACL_INT64,
                inputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                inputDim.data(), inputDim.size(), aData);
            outputTensor = aclCreateTensor(
                outputDim.data(), outputDim.size(), aclDataType::ACL_FLOAT,
                outputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                outputDim.data(), outputDim.size(), cData);
            NlCastType = aclDataType::ACL_FLOAT;
            break;
        case CastType::Uint322Int64:
            inputTensor = aclCreateTensor(
                inputDim.data(), inputDim.size(), aclDataType::ACL_UINT32,
                inputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                inputDim.data(), inputDim.size(), aData);
            outputTensor = aclCreateTensor(
                outputDim.data(), outputDim.size(), aclDataType::ACL_INT64,
                outputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                outputDim.data(), outputDim.size(), cData);
            NlCastType = aclDataType::ACL_INT64;
            break;
        case CastType::Float162Float:
            inputTensor = aclCreateTensor(
                inputDim.data(), inputDim.size(), aclDataType::ACL_FLOAT16,
                inputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                inputDim.data(), inputDim.size(), aData);
            outputTensor = aclCreateTensor(
                outputDim.data(), outputDim.size(), aclDataType::ACL_FLOAT,
                outputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                outputDim.data(), outputDim.size(), cData);
            NlCastType = aclDataType::ACL_FLOAT;
            break;
        case CastType::Float2Float16:
            inputTensor = aclCreateTensor(
                inputDim.data(), inputDim.size(), aclDataType::ACL_FLOAT,
                inputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                inputDim.data(), inputDim.size(), aData);
            outputTensor = aclCreateTensor(
                outputDim.data(), outputDim.size(), aclDataType::ACL_FLOAT16,
                outputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                outputDim.data(), outputDim.size(), cData);
            NlCastType = aclDataType::ACL_FLOAT16;
            break;
        default:
            IT_TODO_HALT();
        }
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;

        auto ret = aclnnCastGetWorkspaceSize(
            inputTensor, NlCastType, outputTensor, &workspaceSize, &executor);
        checkASCENDError(ret);

        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = context->getWorkspace(workspaceSize);
        }

        ret = aclnnCast(workspaceAddr, workspaceSize, executor,
                        context->ASCENDHandle());
        checkASCENDError(ret);

        aclDestroyTensor(inputTensor);
        aclDestroyTensor(outputTensor);
    }
};

REGISTER_KERNEL(Device::ASCEND, OpType::Cast, CastAclnn, "cast_ASCEND");

}; // namespace infini

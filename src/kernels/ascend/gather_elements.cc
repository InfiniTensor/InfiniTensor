#include "operators/gather.h"
#include "aclnnop/aclnn_gather.h"  
#include "ascend/ascend_kernel_without_config.h"
#include "ascend/ascend_runtime.h"

namespace infini {

class GatherElementsAclnn : public ASCENDKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<GatherElementsObj>(_op);
        IT_ASSERT(op->getInputs(1)->getDType() == DataType::Int32 ||
                  op->getInputs(1)->getDType() == DataType::Int64);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);

        void *const data = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const indices = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const output = (op->getOutput()->getRawDataPtr<void *>());

        int64_t axis = int64_t(op->getAxis());

        auto dataDims = op->getInputs(0)->getDims();
        auto dataStride = op->getInputs(0)->getStride();
        auto indicesDims = op->getInputs(1)->getDims();
        auto indicesStride = op->getInputs(1)->getStride();
        auto outputDims = op->getOutput()->getDims();
        auto outputStride = op->getOutput()->getStride();

        std::vector<int64_t> dataDim64 = castTo64(dataDims);
        std::vector<int64_t> dataStride64 = castTo64(dataStride);
        std::vector<int64_t> indicesDim64 = castTo64(indicesDims);
        std::vector<int64_t> indicesStride64 = castTo64(indicesStride);
        std::vector<int64_t> outputDim64 = castTo64(outputDims);
        std::vector<int64_t> outputStride64 = castTo64(outputStride);

        auto aclDataType = aclnnDataTypeConvert(op->getDType());

        auto inputData = aclCreateTensor(
            dataDim64.data(), dataDim64.size(), aclDataType, dataStride64.data(), 0,
            aclFormat::ACL_FORMAT_ND, dataDim64.data(), dataDim64.size(), data);

        auto inputIndices = aclCreateTensor(
            indicesDim64.data(), indicesDim64.size(),
            op->getInputs(1)->getDType() == DataType::Int32 ? ACL_INT32 : ACL_INT64,
            indicesStride64.data(), 0, aclFormat::ACL_FORMAT_ND, indicesDim64.data(),
            indicesDim64.size(), indices);

        auto outputTensor = aclCreateTensor(
            outputDim64.data(), outputDim64.size(), aclDataType, outputStride64.data(), 0,
            aclFormat::ACL_FORMAT_ND, outputDim64.data(), outputDim64.size(), output);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;

        auto ret = aclnnGatherGetWorkspaceSize(inputData, axis, inputIndices, outputTensor,
                                                       &workspaceSize, &executor);
        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = context->getWorkspace(workspaceSize);
        }
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("aclnnGatherGetWorkspaceSize failed. ERROR: %d\n",
                            ret));

        ret = aclnnGather(workspaceAddr, workspaceSize, executor,
                                  context->ASCENDHandle());
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("aclnnGather failed. ERROR: %d\n", ret));

        ret = aclrtSynchronizeStream(context->ASCENDHandle());
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret));
        return;
    }
};

REGISTER_KERNEL(Device::ASCEND, OpType::GatherElements, GatherElementsAclnn,
                "gather_elements_ASCEND_float");

}; // namespace infini

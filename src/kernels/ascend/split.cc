#include "operators/split.h"
#include "aclnnop/aclnn_split_with_size.h"
#include "ascend/ascend_kernel_without_config.h"
#include "ascend/ascend_runtime.h"

namespace infini {

class SplitAclnn : public ASCENDKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<SplitObj>(_op);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());

        auto a = op->getInputs(0)->getDims();
        auto aS = op->getInputs(0)->getStride();
        std::vector<int64_t> aDim = castTo64(a);
        std::vector<int64_t> aStride = castTo64(aS);

        int64_t dim = op->getDim();
        int num = op->numOutputs();
        // int dimSize = a.at(op->getDim());
        // uint64_t splitSections = dimSize / num;
        vector<int> ratio = op->getRatio();
        std::vector<int64_t> Ratio = castTo64(ratio);

        auto aclDataType = aclnnDataTypeConvert(op->getDType());

        auto inputA = aclCreateTensor(
            aDim.data(), aDim.size(), aclDataType, aStride.data(), 0,
            aclFormat::ACL_FORMAT_ND, aDim.data(), aDim.size(), aData);

        std::vector<aclTensor *> outputsData{};
        for (int i = 0; i < num; ++i) {
            auto c = op->getOutput(i)->getDims();
            auto cS = op->getOutput(i)->getStride();

            std::vector<int64_t> cDim = castTo64(c);
            std::vector<int64_t> cStride = castTo64(cS);

            void *const cData = (op->getOutput(i)->getRawDataPtr<void *>());

            aclTensor *tmpTensor = aclCreateTensor(
                cDim.data(), cDim.size(), aclDataType, cStride.data(), 0,
                aclFormat::ACL_FORMAT_ND, cDim.data(), cDim.size(), cData);

            outputsData.push_back(tmpTensor);
        }
        aclTensorList *tensorList =
            aclCreateTensorList(outputsData.data(), outputsData.size());

        aclIntArray *splitSize = aclCreateIntArray(Ratio.data(), Ratio.size());

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;

        // auto ret = aclnnSplitTensorGetWorkspaceSize(
        //     inputA, splitSections, dim, tensorList, &workspaceSize,
        //     &executor);
        auto ret = aclnnSplitWithSizeGetWorkspaceSize(
            inputA, splitSize, dim, tensorList, &workspaceSize, &executor);
        checkASCENDError(ret);

        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = context->getWorkspace(workspaceSize);
        }

        // ret = aclnnSplitTensor(workspaceAddr, workspaceSize, executor,
        //                        context->ASCENDHandle());
        ret = aclnnSplitWithSize(workspaceAddr, workspaceSize, executor,
                                 context->ASCENDHandle());
        checkASCENDError(ret);

        aclDestroyTensor(inputA);
        aclDestroyTensorList(tensorList);

        return;
    }
};

REGISTER_KERNEL(Device::ASCEND, OpType::Split, SplitAclnn,
                "split_ASCEND_float");
} // namespace infini

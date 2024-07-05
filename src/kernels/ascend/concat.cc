#include "operators/concat.h"
#include "aclnnop/level2/aclnn_cat.h"
#include "ascend/ascend_kernel_without_config.h"
#include "ascend/ascend_runtime.h"

namespace infini {

class ConcatAclnn : public ASCENDKernelWithoutConfig {

    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ConcatObj>(_op);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);
        IT_ASSERT(op->getDType() == DataType::Float32);

        int dim = op->getDim();
        int num = op->numInputs();

        std::vector<aclTensor *> inputsData{};

        for (int i = 0; i < num; ++i) {
            auto inD = op->getInputs(i)->getDims();
            auto inS = op->getInputs(i)->getStride();
            std::vector<int64_t> inputDim = castTo64(inD);
            std::vector<int64_t> inputStride = castTo64(inS);

            void *const inData = (op->getInputs(i)->getRawDataPtr<void *>());
            auto tmpTensor =
                aclCreateTensor(inputDim.data(), inputDim.size(), ACL_FLOAT,
                                inputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                                inputDim.data(), inputDim.size(), inData);

            inputsData.push_back(tmpTensor);
        }
        aclTensorList *tensorList =
            aclCreateTensorList(inputsData.data(), inputsData.size());

        void *const outData = (op->getOutput()->getRawDataPtr<void *>());
        auto outD = op->getOutput()->getDims();
        auto outS = op->getOutput()->getStride();
        std::vector<int64_t> outputDim = castTo64(outD);
        std::vector<int64_t> outputStride = castTo64(outS);

        auto outputTensor =
            aclCreateTensor(outputDim.data(), outputDim.size(), ACL_FLOAT,
                            outputStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                            outputDim.data(), outputDim.size(), outData);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;

        auto ret = aclnnCatGetWorkspaceSize(
            tensorList, int64_t(dim), outputTensor, &workspaceSize, &executor);
        checkASCENDError(ret);

        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = context->getWorkspace(workspaceSize);
        }

        ret = aclnnCat(workspaceAddr, workspaceSize, executor,
                       context->ASCENDHandle());
        checkASCENDError(ret);

        aclDestroyTensorList(tensorList);
        aclDestroyTensor(outputTensor);

        return;
    }
};

REGISTER_KERNEL(Device::ASCEND, OpType::Concat, ConcatAclnn,
                "concat_ASCEND_float");
} // namespace infini

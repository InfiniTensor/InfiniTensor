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
        int dim = op->getDim();
        //int num = op->numInputs();


	std::vector<aclTensor*> inputsData{};
        auto inD0 = op->getInputs(0)->getDims();
        auto inS0 = op->getInputs(0)->getStride();
        std::vector<int64_t> inputDim0 = MycastTo64(inD0);
        std::vector<int64_t> inputStride0 = MycastTo64(inS0);

        void *const inData0 = (op->getInputs(0)->getRawDataPtr<void *>());
        auto tmpTensor0 = aclCreateTensor(
        inputDim0.data(), inputDim0.size(), ACL_FLOAT, inputStride0.data(), 0,
        aclFormat::ACL_FORMAT_ND, inputDim0.data(), inputDim0.size(), inData0);

        inputsData.push_back(tmpTensor0);

        auto inD = op->getInputs(1)->getDims();
        auto inS = op->getInputs(1)->getStride();
        std::vector<int64_t> inputDim = MycastTo64(inD);
        std::vector<int64_t> inputStride = MycastTo64(inS);

        void *const inData = (op->getInputs(1)->getRawDataPtr<void *>());
        auto tmpTensor = aclCreateTensor(
        inputDim.data(), inputDim.size(), ACL_FLOAT, inputStride.data(), 0,
        aclFormat::ACL_FORMAT_ND, inputDim.data(), inputDim.size(), inData);

        inputsData.push_back(tmpTensor);
        //for (int i = 0; i < num; ++i) {
        //    auto inD = op->getInputs(i)->getDims();
        //    auto inS = op->getInputs(i)->getStride();
        //    std::vector<int64_t> inputDim = MycastTo64(inD);
        //    std::vector<int64_t> inputStride = MycastTo64(inS);

        //    void *const inData = (op->getInputs(i)->getRawDataPtr<void *>());
        //    auto tmpTensor = aclCreateTensor(
        //    inputDim.data(), inputDim.size(), ACL_FLOAT, inputStride.data(), 0,
        //    aclFormat::ACL_FORMAT_ND, inputDim.data(), inputDim.size(), inData);

        //    inputsData.push_back(tmpTensor);
        //}
	aclTensorList* tensorList = aclCreateTensorList(inputsData.data(), inputsData.size());

        void *const outData = (op->getOutput()->getRawDataPtr<void *>());
        auto outD = op->getOutput()->getDims();
        auto outS = op->getOutput()->getStride();
        std::vector<int64_t> outputDim = MycastTo64(outD);
        std::vector<int64_t> outputStride = MycastTo64(outS);

        auto outputTensor = aclCreateTensor(
            outputDim.data(), outputDim.size(), ACL_FLOAT, outputStride.data(), 0,
            aclFormat::ACL_FORMAT_ND, outputDim.data(), outputDim.size(), outData);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;

        auto ret =
            aclnnCatGetWorkspaceSize(tensorList, int64_t(dim), outputTensor, &workspaceSize, &executor);
        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize,
                              ACL_MEM_MALLOC_HUGE_FIRST);
        }
        assert(ret == ACL_SUCCESS);
        ret = aclnnCat(workspaceAddr, workspaceSize, executor,
                        context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);
	
        ret = aclrtSynchronizeStream(context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);

	aclDestroyTensorList(tensorList);
	aclDestroyTensor(outputTensor);

        return;
    }
};





REGISTER_KERNEL(Device::ASCEND, OpType::Concat, DataType::Float32, ConcatAclnn,
                "concat_ASCEND_float");
}; // namespace infini

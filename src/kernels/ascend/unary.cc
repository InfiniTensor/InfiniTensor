#include "operators/unary.h"
#include "ascend/ascend_kernel_without_config.h"
#include "ascend/ascend_runtime.h"
#include "aclnnop/level2/aclnn_relu.h"

namespace infini {
class ReluAclnn : public ASCENDKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

	auto a = op->getInputs(0)->getDims();
	std::vector<int64_t> aDim(a.size(), 1);
	for(size_t i = 0; i < a.size(); ++i) {
	  aDim[i] = int64_t(a[i]); 
	}
	auto aS = op->getInputs(0)->getStride();
	std::vector<int64_t> aStride(aS.size(), 1);
	for(size_t i = 0; i < aS.size(); ++i) {
	  aStride[i] = int64_t(aS[i]); 
	}
	auto c = op->getInputs(0)->getDims();
	std::vector<int64_t> cDim(c.size(), 1);
	for(size_t i = 0; i < c.size(); ++i) {
	  cDim[i] = int64_t(c[i]); 
	}
	auto cS = op->getInputs(0)->getStride();
	std::vector<int64_t> cStride(cS.size(), 1);
	for(size_t i = 0; i < cS.size(); ++i) {
	  cStride[i] = int64_t(cS[i]); 
	}

	auto input = aclCreateTensor(aDim.data(), aDim.size(), ACL_FLOAT, aStride.data(), 0, aclFormat::ACL_FORMAT_ND, aDim.data(), aDim.size(), aData);
	auto output = aclCreateTensor(cDim.data(), cDim.size(), ACL_FLOAT, cStride.data(), 0, aclFormat::ACL_FORMAT_ND, cDim.data(), cDim.size(), cData);

	uint64_t workspaceSize = 0;
	aclOpExecutor* executor;

	auto ret = aclnnReluGetWorkspaceSize(input, output, &workspaceSize, &executor);
	void* workspaceAddr = nullptr;
  	if (workspaceSize > 0) {
		ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
  	}
        assert(ret == ACL_SUCCESS);
	ret = aclnnRelu(workspaceAddr, workspaceSize, executor, context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);
	ret = aclrtSynchronizeStream(context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);

        return;
    }
};
}

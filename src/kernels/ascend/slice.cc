#include "operators/slice.h"
#include "aclnnop/aclnn_slice_v2.h"
#include "ascend/ascend_kernel_without_config.h"
#include "ascend/ascend_runtime.h"

namespace infini {

class SliceAclnn : public ASCENDKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<SliceObj>(_op);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto starts_32 = op->getStarts();
        auto ends_32 = op->getEnds();
        auto steps_32 = op->getSteps();

        auto a = op->getInputs(0)->getDims();
        auto aS = op->getInputs(0)->getStride();
        auto c = op->getOutput()->getDims();
        auto cS = op->getOutput()->getStride();

        std::vector<int64_t> aDim = castTo64(a);
        std::vector<int64_t> aStride = castTo64(aS);
        std::vector<int64_t> cDim = castTo64(c);
        std::vector<int64_t> cStride = castTo64(cS);

        std::vector<int64_t> starts_64 = castTo64(starts_32);
        std::vector<int64_t> ends_64 = castTo64(ends_32);
        std::vector<int64_t> steps_64 = castTo64(steps_32);

        vector<int64_t> axes_64 = vector<int64_t>(starts_32.size(), 0);
        for (int i = 0; i < int(starts_32.size()); i++) {
            axes_64[i] = i;
        }

        auto aclDataType = aclnnDataTypeConvert(op->getDType());

        auto inputA = aclCreateTensor(
            aDim.data(), aDim.size(), aclDataType, aStride.data(), 0,
            aclFormat::ACL_FORMAT_ND, aDim.data(), aDim.size(), aData);
        auto output = aclCreateTensor(
            cDim.data(), cDim.size(), aclDataType, cStride.data(), 0,
            aclFormat::ACL_FORMAT_ND, cDim.data(), cDim.size(), cData);
        aclIntArray *starts =
            aclCreateIntArray(starts_64.data(), starts_64.size());
        aclIntArray *ends = aclCreateIntArray(ends_64.data(), ends_64.size());
        aclIntArray *steps =
            aclCreateIntArray(steps_64.data(), steps_64.size());
        aclIntArray *axes = aclCreateIntArray(axes_64.data(), axes_64.size());

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;

        auto ret =
            aclnnSliceV2GetWorkspaceSize(inputA, starts, ends, axes, steps,
                                         output, &workspaceSize, &executor);
        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = context->getWorkspace(workspaceSize);
        }
        assert(ret == ACL_SUCCESS);
        ret = aclnnSliceV2(workspaceAddr, workspaceSize, executor,
                           context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);

        return;
    }
};

REGISTER_KERNEL(Device::ASCEND, OpType::Slice, SliceAclnn,
                "slice_ASCEND_float");
}; // namespace infini

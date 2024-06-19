#include "operators/element_wise.h"
#include "aclnnop/aclnn_maximum.h"
#include "aclnnop/level2/aclnn_add.h"
#include "aclnnop/level2/aclnn_div.h"
#include "aclnnop/level2/aclnn_mul.h"
#include "aclnnop/level2/aclnn_pow_tensor_tensor.h"
#include "aclnnop/level2/aclnn_sub.h"
#include "ascend/ascend_kernel_without_config.h"
#include "ascend/ascend_runtime.h"

namespace infini {

#define DEFINE_ELEMENT_WISE_Aclnn(prefix)                                      \
    class prefix##Aclnn : public ASCENDKernelWithoutConfig {                   \
        void compute(const Operator &_op,                                      \
                     const RuntimeObj *_context) const override {              \
            auto op = as<ElementWiseObj>(_op);                                 \
            auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);   \
                                                                               \
            void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());   \
            void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());   \
            void *const cData = (op->getOutput()->getRawDataPtr<void *>());    \
                                                                               \
            auto a = op->getInputs(0)->getDims();                              \
            auto aS = op->getInputs(0)->getStride();                           \
            auto b = op->getInputs(1)->getDims();                              \
            auto bS = op->getInputs(1)->getStride();                           \
            auto c = op->getOutput()->getDims();                               \
            auto cS = op->getOutput()->getStride();                            \
                                                                               \
            std::vector<int64_t> aDim = castTo64(a);                           \
            std::vector<int64_t> aStride = castTo64(aS);                       \
            std::vector<int64_t> bDim = castTo64(b);                           \
            std::vector<int64_t> bStride = castTo64(bS);                       \
            std::vector<int64_t> cDim = castTo64(c);                           \
            std::vector<int64_t> cStride = castTo64(cS);                       \
                                                                               \
            auto aclDataType = aclnnDataTypeConvert(op->getDType());           \
                                                                               \
            auto inputA = aclCreateTensor(                                     \
                aDim.data(), aDim.size(), aclDataType, aStride.data(), 0,      \
                aclFormat::ACL_FORMAT_ND, aDim.data(), aDim.size(), aData);    \
            auto inputB = aclCreateTensor(                                     \
                bDim.data(), bDim.size(), aclDataType, bStride.data(), 0,      \
                aclFormat::ACL_FORMAT_ND, bDim.data(), bDim.size(), bData);    \
            auto output = aclCreateTensor(                                     \
                cDim.data(), cDim.size(), aclDataType, cStride.data(), 0,      \
                aclFormat::ACL_FORMAT_ND, cDim.data(), cDim.size(), cData);    \
                                                                               \
            uint64_t workspaceSize = 0;                                        \
            aclOpExecutor *executor;                                           \
                                                                               \
            auto ret = aclnn##prefix##GetWorkspaceSize(                        \
                inputA, inputB, output, &workspaceSize, &executor);            \
            void *workspaceAddr = nullptr;                                     \
            if (workspaceSize > 0) {                                           \
                workspaceAddr = context->getWorkspace(workspaceSize);          \
            }                                                                  \
            assert(ret == ACL_SUCCESS);                                        \
            ret = aclnn##prefix(workspaceAddr, workspaceSize, executor,        \
                                context->ASCENDHandle());                      \
            assert(ret == ACL_SUCCESS);                                        \
                                                                               \
            ret = aclDestroyTensor(inputA);                                    \
            ret = aclDestroyTensor(inputB);                                    \
            ret = aclDestroyTensor(output);                                    \
                                                                               \
            return;                                                            \
        }                                                                      \
    };

class AddAclnn : public ASCENDKernelWithoutConfig {
    virtual tuple<float, float, float> getAlphBeta() const {
        return {1.f, 1.f, 0.f};
    }
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto a = op->getInputs(0)->getDims();
        auto aS = op->getInputs(0)->getStride();
        auto b = op->getInputs(1)->getDims();
        auto bS = op->getInputs(1)->getStride();
        auto c = op->getOutput()->getDims();
        auto cS = op->getOutput()->getStride();

        std::vector<int64_t> aDim = castTo64(a);
        std::vector<int64_t> aStride = castTo64(aS);
        std::vector<int64_t> bDim = castTo64(b);
        std::vector<int64_t> bStride = castTo64(bS);
        std::vector<int64_t> cDim = castTo64(c);
        std::vector<int64_t> cStride = castTo64(cS);

        auto aclDataType = aclnnDataTypeConvert(op->getDType());
        
        auto inputA = aclCreateTensor(
            aDim.data(), aDim.size(), aclDataType, aStride.data(), 0,
            aclFormat::ACL_FORMAT_ND, aDim.data(), aDim.size(), aData);
        auto inputB = aclCreateTensor(
            bDim.data(), bDim.size(), aclDataType, bStride.data(), 0,
            aclFormat::ACL_FORMAT_ND, bDim.data(), bDim.size(), bData);
        auto output = aclCreateTensor(
            cDim.data(), cDim.size(), aclDataType, cStride.data(), 0,
            aclFormat::ACL_FORMAT_ND, cDim.data(), cDim.size(), cData);

        auto [aAlpha, bAlpha, beta] = getAlphBeta();
        // ACL_FLOAT can be converted to ACL_FLOAT16 automatically
        auto alpha = aclCreateScalar(&bAlpha, ACL_FLOAT);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;

        auto ret = aclnnAddGetWorkspaceSize(inputA, inputB, alpha, output,
                                            &workspaceSize, &executor);
        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = context->getWorkspace(workspaceSize);
        }
        assert(ret == ACL_SUCCESS);
        ret = aclnnAdd(workspaceAddr, workspaceSize, executor,
                       context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);

        return;
    }
};

class SubAclnn : public ASCENDKernelWithoutConfig {
    virtual tuple<float, float, float> getAlphBeta() const {
        return {1.f, 1.f, 0.f};
    }
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto a = op->getInputs(0)->getDims();
        auto aS = op->getInputs(0)->getStride();
        auto b = op->getInputs(1)->getDims();
        auto bS = op->getInputs(1)->getStride();
        auto c = op->getOutput()->getDims();
        auto cS = op->getOutput()->getStride();

        std::vector<int64_t> aDim = castTo64(a);
        std::vector<int64_t> aStride = castTo64(aS);
        std::vector<int64_t> bDim = castTo64(b);
        std::vector<int64_t> bStride = castTo64(bS);
        std::vector<int64_t> cDim = castTo64(c);
        std::vector<int64_t> cStride = castTo64(cS);

        auto aclDataType = aclnnDataTypeConvert(op->getDType());

        auto inputA = aclCreateTensor(
            aDim.data(), aDim.size(), aclDataType, aStride.data(), 0,
            aclFormat::ACL_FORMAT_ND, aDim.data(), aDim.size(), aData);
        auto inputB = aclCreateTensor(
            bDim.data(), bDim.size(), aclDataType, bStride.data(), 0,
            aclFormat::ACL_FORMAT_ND, bDim.data(), bDim.size(), bData);
        auto output = aclCreateTensor(
            cDim.data(), cDim.size(), aclDataType, cStride.data(), 0,
            aclFormat::ACL_FORMAT_ND, cDim.data(), cDim.size(), cData);

        auto [aAlpha, bAlpha, beta] = getAlphBeta();
        // ACL_FLOAT can be converted to ACL_FLOAT16 automatically
        auto alpha = aclCreateScalar(&bAlpha, ACL_FLOAT);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;

        auto ret = aclnnSubGetWorkspaceSize(inputA, inputB, alpha, output,
                                            &workspaceSize, &executor);
        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = context->getWorkspace(workspaceSize);
        }
        assert(ret == ACL_SUCCESS);
        ret = aclnnSub(workspaceAddr, workspaceSize, executor,
                       context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);

        ret = aclDestroyTensor(inputA);
        ret = aclDestroyTensor(inputB);
        ret = aclDestroyScalar(alpha);
        ret = aclDestroyTensor(output);

        return;
    }
};

DEFINE_ELEMENT_WISE_Aclnn(PowTensorTensor);
DEFINE_ELEMENT_WISE_Aclnn(Div);
DEFINE_ELEMENT_WISE_Aclnn(Mul);
DEFINE_ELEMENT_WISE_Aclnn(Maximum);

REGISTER_KERNEL(Device::ASCEND, OpType::Pow, PowTensorTensorAclnn,
                "pow_ASCEND_float");
REGISTER_KERNEL(Device::ASCEND, OpType::Div, DivAclnn, "div_ASCEND_float");
REGISTER_KERNEL(Device::ASCEND, OpType::Mul, MulAclnn, "mul_ASCEND_float");

REGISTER_KERNEL(Device::ASCEND, OpType::Add, AddAclnn, "add_ASCEND_float");
REGISTER_KERNEL(Device::ASCEND, OpType::Sub, SubAclnn, "sub_ASCEND_float");
REGISTER_KERNEL(Device::ASCEND, OpType::Max, MaximumAclnn, "max_ASCEND_float");
//  REGISTER_KERNEL(Device::ASCEND, OpType::Abs, AbsAclnn, "abs_ASCEND_float");

}; // namespace infini

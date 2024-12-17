#include "operators/element_wise.h"
#include "aclnnop/aclnn_maximum.h"
#include "aclnnop/level2/aclnn_add.h"
#include "aclnnop/level2/aclnn_div.h"
#include "aclnnop/level2/aclnn_eq_tensor.h"
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
            IT_ASSERT(op->getDType() == DataType::Float32);                    \
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
            checkASCENDError(ret);                                             \
            void *workspaceAddr = nullptr;                                     \
            if (workspaceSize > 0) {                                           \
                workspaceAddr = context->getWorkspace(workspaceSize);          \
            }                                                                  \
                                                                               \
            ret = aclnn##prefix(workspaceAddr, workspaceSize, executor,        \
                                context->ASCENDHandle());                      \
            checkASCENDError(ret);                                             \
                                                                               \
            aclDestroyTensor(inputA);                                          \
            aclDestroyTensor(inputB);                                          \
            aclDestroyTensor(output);                                          \
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
        IT_ASSERT(op->getDType() == DataType::Float32);

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
        checkASCENDError(ret);

        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = context->getWorkspace(workspaceSize);
        }

        ret = aclnnAdd(workspaceAddr, workspaceSize, executor,
                       context->ASCENDHandle());
        checkASCENDError(ret);

        aclDestroyTensor(inputA);
        aclDestroyTensor(inputB);
        aclDestroyScalar(alpha);
        aclDestroyTensor(output);

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
        IT_ASSERT(op->getDType() == DataType::Float32);

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
        checkASCENDError(ret);

        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = context->getWorkspace(workspaceSize);
        }

        ret = aclnnSub(workspaceAddr, workspaceSize, executor,
                       context->ASCENDHandle());
        checkASCENDError(ret);

        aclDestroyTensor(inputA);
        aclDestroyTensor(inputB);
        aclDestroyScalar(alpha);
        aclDestroyTensor(output);

        return;
    }
};

DEFINE_ELEMENT_WISE_Aclnn(PowTensorTensor);
DEFINE_ELEMENT_WISE_Aclnn(Div);
DEFINE_ELEMENT_WISE_Aclnn(Mul);
DEFINE_ELEMENT_WISE_Aclnn(Maximum);

DEFINE_ELEMENT_WISE_Aclnn(EqTensor);

REGISTER_KERNEL(Device::ASCEND, OpType::Pow, PowTensorTensorAclnn,
                "pow_ASCEND_float");
REGISTER_KERNEL(Device::ASCEND, OpType::Div, DivAclnn, "div_ASCEND_float");
REGISTER_KERNEL(Device::ASCEND, OpType::Mul, MulAclnn, "mul_ASCEND_float");

REGISTER_KERNEL(Device::ASCEND, OpType::Add, AddAclnn, "add_ASCEND_float");
REGISTER_KERNEL(Device::ASCEND, OpType::Sub, SubAclnn, "sub_ASCEND_float");
REGISTER_KERNEL(Device::ASCEND, OpType::Max, MaximumAclnn, "max_ASCEND_float");
//  REGISTER_KERNEL(Device::ASCEND, OpType::Abs, AbsAclnn, "abs_ASCEND_float");

REGISTER_KERNEL(Device::ASCEND, OpType::Equal, EqTensorAclnn, "equal_ASCEND");
// REGISTER_KERNEL(Device::BANG, OpType::Greater, GreaterThanCnnl,
//                 "GreaterThan_cnnl_BANG");
// REGISTER_KERNEL(Device::BANG, OpType::GreaterOrEqual, GreaterEqualCnnl,
//                 "GreaterEqual_cnnl_BANG");
// REGISTER_KERNEL(Device::BANG, OpType::Less, LessThanCnnl,
// "LessThan_cnnl_BANG"); REGISTER_KERNEL(Device::BANG, OpType::LessOrEqual,
// LessEqualCnnl,
//                 "LessEqual_cnnl_BANG");

} // namespace infini

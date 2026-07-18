#include "operators/unary.h"
#include "aclnnop/aclnn_erf.h"
#include "aclnnop/level2/aclnn_abs.h"
#include "aclnnop/level2/aclnn_acos.h"
#include "aclnnop/level2/aclnn_atan.h"
#include "aclnnop/level2/aclnn_ceil.h"
#include "aclnnop/level2/aclnn_cos.h"
#include "aclnnop/level2/aclnn_exp.h"
#include "aclnnop/level2/aclnn_floor.h"
#include "aclnnop/level2/aclnn_gelu.h"
#include "aclnnop/level2/aclnn_hardswish.h"
#include "aclnnop/level2/aclnn_leaky_relu.h"
#include "aclnnop/level2/aclnn_neg.h"
#include "aclnnop/level2/aclnn_reciprocal.h"
#include "aclnnop/level2/aclnn_relu.h"
#include "aclnnop/level2/aclnn_round.h"
#include "aclnnop/level2/aclnn_sigmoid.h"
#include "aclnnop/level2/aclnn_hardsigmoid.h"
#include "aclnnop/level2/aclnn_sin.h"
#include "aclnnop/level2/aclnn_sqrt.h"
#include "aclnnop/level2/aclnn_tanh.h"
#include "ascend/ascend_kernel_without_config.h"
#include "ascend/ascend_runtime.h"

namespace infini {
class ReluAclnn : public ASCENDKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);
        IT_ASSERT(op->getDType() == DataType::Float32);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto a = op->getInputs(0)->getDims();
        std::vector<int64_t> aDim(a.size(), 1);
        for (size_t i = 0; i < a.size(); ++i) {
            aDim[i] = int64_t(a[i]);
        }
        auto aS = op->getInputs(0)->getStride();
        std::vector<int64_t> aStride(aS.size(), 1);
        for (size_t i = 0; i < aS.size(); ++i) {
            aStride[i] = int64_t(aS[i]);
        }
        auto c = op->getInputs(0)->getDims();
        std::vector<int64_t> cDim(c.size(), 1);
        for (size_t i = 0; i < c.size(); ++i) {
            cDim[i] = int64_t(c[i]);
        }
        auto cS = op->getInputs(0)->getStride();
        std::vector<int64_t> cStride(cS.size(), 1);
        for (size_t i = 0; i < cS.size(); ++i) {
            cStride[i] = int64_t(cS[i]);
        }

        auto input = aclCreateTensor(
            aDim.data(), aDim.size(), ACL_FLOAT, aStride.data(), 0,
            aclFormat::ACL_FORMAT_ND, aDim.data(), aDim.size(), aData);
        auto output = aclCreateTensor(
            cDim.data(), cDim.size(), ACL_FLOAT, cStride.data(), 0,
            aclFormat::ACL_FORMAT_ND, cDim.data(), cDim.size(), cData);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;

        auto ret =
            aclnnReluGetWorkspaceSize(input, output, &workspaceSize, &executor);
        checkASCENDError(ret);

        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = context->getWorkspace(workspaceSize);
        }

        ret = aclnnRelu(workspaceAddr, workspaceSize, executor,
                        context->ASCENDHandle());
        checkASCENDError(ret);

        aclDestroyTensor(input);
        aclDestroyTensor(output);

        return;
    }
};
class LeakyReluAclnn : public ASCENDKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<LeakyReluObj>(_op);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);
        IT_ASSERT(op->getDType() == DataType::Float32);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto a = op->getInputs(0)->getDims();
        std::vector<int64_t> aDim(a.size(), 1);
        for (size_t i = 0; i < a.size(); ++i) {
            aDim[i] = int64_t(a[i]);
        }
        auto aS = op->getInputs(0)->getStride();
        std::vector<int64_t> aStride(aS.size(), 1);
        for (size_t i = 0; i < aS.size(); ++i) {
            aStride[i] = int64_t(aS[i]);
        }
        auto c = op->getInputs(0)->getDims();
        std::vector<int64_t> cDim(c.size(), 1);
        for (size_t i = 0; i < c.size(); ++i) {
            cDim[i] = int64_t(c[i]);
        }
        auto cS = op->getInputs(0)->getStride();
        std::vector<int64_t> cStride(cS.size(), 1);
        for (size_t i = 0; i < cS.size(); ++i) {
            cStride[i] = int64_t(cS[i]);
        }

        auto input = aclCreateTensor(
            aDim.data(), aDim.size(), ACL_FLOAT, aStride.data(), 0,
            aclFormat::ACL_FORMAT_ND, aDim.data(), aDim.size(), aData);
        auto output = aclCreateTensor(
            cDim.data(), cDim.size(), ACL_FLOAT, cStride.data(), 0,
            aclFormat::ACL_FORMAT_ND, cDim.data(), cDim.size(), cData);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;
        float negativeSlopeValue = op->getAlpha();
        aclScalar *negativeSlope = nullptr;
        negativeSlope =
            aclCreateScalar(&negativeSlopeValue, aclDataType::ACL_FLOAT);

        auto ret = aclnnLeakyReluGetWorkspaceSize(input, negativeSlope, output,
                                                  &workspaceSize, &executor);
        checkASCENDError(ret);

        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = context->getWorkspace(workspaceSize);
        }

        ret = aclnnLeakyRelu(workspaceAddr, workspaceSize, executor,
                             context->ASCENDHandle());
        checkASCENDError(ret);

        aclDestroyTensor(input);
        aclDestroyScalar(negativeSlope);
        aclDestroyTensor(output);

        return;
    }
};

#define DEFINE_UNARY_Aclnn(prefix)                                             \
    class prefix##Aclnn : public ASCENDKernelWithoutConfig {                   \
        void compute(const Operator &_op,                                      \
                     const RuntimeObj *_context) const override {              \
            auto op = as<UnaryObj>(_op);                                       \
            auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);   \
            IT_ASSERT(op->getDType() == DataType::Float32);                    \
                                                                               \
            void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());   \
            void *const cData = (op->getOutput()->getRawDataPtr<void *>());    \
                                                                               \
            auto a = op->getInputs(0)->getDims();                              \
            std::vector<int64_t> aDim(a.size(), 1);                            \
            for (size_t i = 0; i < a.size(); ++i) {                            \
                aDim[i] = int64_t(a[i]);                                       \
            }                                                                  \
            auto aS = op->getInputs(0)->getStride();                           \
            std::vector<int64_t> aStride(aS.size(), 1);                        \
            for (size_t i = 0; i < aS.size(); ++i) {                           \
                aStride[i] = int64_t(aS[i]);                                   \
            }                                                                  \
            auto c = op->getInputs(0)->getDims();                              \
            std::vector<int64_t> cDim(c.size(), 1);                            \
            for (size_t i = 0; i < c.size(); ++i) {                            \
                cDim[i] = int64_t(c[i]);                                       \
            }                                                                  \
            auto cS = op->getInputs(0)->getStride();                           \
            std::vector<int64_t> cStride(cS.size(), 1);                        \
            for (size_t i = 0; i < cS.size(); ++i) {                           \
                cStride[i] = int64_t(cS[i]);                                   \
            }                                                                  \
                                                                               \
            auto input = aclCreateTensor(                                      \
                aDim.data(), aDim.size(), ACL_FLOAT, aStride.data(), 0,        \
                aclFormat::ACL_FORMAT_ND, aDim.data(), aDim.size(), aData);    \
            auto output = aclCreateTensor(                                     \
                cDim.data(), cDim.size(), ACL_FLOAT, cStride.data(), 0,        \
                aclFormat::ACL_FORMAT_ND, cDim.data(), cDim.size(), cData);    \
                                                                               \
            uint64_t workspaceSize = 0;                                        \
            aclOpExecutor *executor;                                           \
                                                                               \
            auto ret = aclnn##prefix##GetWorkspaceSize(                        \
                input, output, &workspaceSize, &executor);                     \
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
            aclDestroyTensor(input);                                           \
            aclDestroyTensor(output);                                          \
                                                                               \
            return;                                                            \
        }                                                                      \
    };

DEFINE_UNARY_Aclnn(Abs);
DEFINE_UNARY_Aclnn(Sigmoid);
DEFINE_UNARY_Aclnn(Hardsigmoid);
DEFINE_UNARY_Aclnn(Hardswish);
DEFINE_UNARY_Aclnn(Gelu);

DEFINE_UNARY_Aclnn(Tanh);
DEFINE_UNARY_Aclnn(Sin);
DEFINE_UNARY_Aclnn(Cos);
DEFINE_UNARY_Aclnn(Acos);
DEFINE_UNARY_Aclnn(Atan);

DEFINE_UNARY_Aclnn(Ceil);
DEFINE_UNARY_Aclnn(Floor);
DEFINE_UNARY_Aclnn(Exp);
DEFINE_UNARY_Aclnn(Neg);
DEFINE_UNARY_Aclnn(Reciprocal);
DEFINE_UNARY_Aclnn(Sqrt);
DEFINE_UNARY_Aclnn(Round);

DEFINE_UNARY_Aclnn(Erf);

REGISTER_KERNEL(Device::ASCEND, OpType::Relu, ReluAclnn, "relu_ASCEND_float");
REGISTER_KERNEL(Device::ASCEND, OpType::LeakyRelu, LeakyReluAclnn,
                "leakyrelu_ASCEND_float");
REGISTER_KERNEL(Device::ASCEND, OpType::Abs, AbsAclnn, "abs_ASCEND_float");
REGISTER_KERNEL(Device::ASCEND, OpType::Sigmoid, SigmoidAclnn,
                "sigmoid_ASCEND_float");
// HardSigmoid has different def in acl and onnx, we use acl here.
// TODO: Replace this acl def with onnx def
REGISTER_KERNEL(Device::ASCEND, OpType::HardSigmoid, HardsigmoidAclnn,
                "hardsigmoid_ASCEND_float");                
REGISTER_KERNEL(Device::ASCEND, OpType::HardSwish, HardswishAclnn,
                "hardswish_ASCEND_float");
REGISTER_KERNEL(Device::ASCEND, OpType::Tanh, TanhAclnn, "tanh_ASCEND_float");
REGISTER_KERNEL(Device::ASCEND, OpType::Gelu, GeluAclnn, "gelu_ASCEND_float");
REGISTER_KERNEL(Device::ASCEND, OpType::Sin, SinAclnn, "sin_ASCEND_float");
REGISTER_KERNEL(Device::ASCEND, OpType::Cos, CosAclnn, "cos_ASCEND_float");
REGISTER_KERNEL(Device::ASCEND, OpType::Acos, AcosAclnn, "acos_ASCEND_float");
REGISTER_KERNEL(Device::ASCEND, OpType::Atan, AtanAclnn, "atan_ASCEND_float");
REGISTER_KERNEL(Device::ASCEND, OpType::Neg, NegAclnn, "neg_ASCEND_float");
REGISTER_KERNEL(Device::ASCEND, OpType::Ceil, CeilAclnn, "ceil_ASCEND_float");
REGISTER_KERNEL(Device::ASCEND, OpType::Floor, FloorAclnn,
                "floor_ASCEND_float");
REGISTER_KERNEL(Device::ASCEND, OpType::Exp, ExpAclnn, "exp_ASCEND_float");
REGISTER_KERNEL(Device::ASCEND, OpType::Reciprocal, ReciprocalAclnn,
                "reciprocal_ASCEND_float");
REGISTER_KERNEL(Device::ASCEND, OpType::Sqrt, SqrtAclnn, "sqrt_ASCEND_float");
REGISTER_KERNEL(Device::ASCEND, OpType::Round, RoundAclnn,
                "round_ASCEND_float");
REGISTER_KERNEL(Device::ASCEND, OpType::Erf, ErfAclnn, "erf_ASCEND_float");
} // namespace infini

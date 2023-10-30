#include "operators/unary.h"
#include "aclnnop/level2/aclnn_relu.h"
#include "aclnnop/level2/aclnn_abs.h"
#include "aclnnop/level2/aclnn_sigmoid.h"
#include "aclnnop/level2/aclnn_hardswish.h"
#include "aclnnop/level2/aclnn_tanh.h"
#include "aclnnop/level2/aclnn_gelu.h"
#include "aclnnop/level2/aclnn_sin.h"
#include "aclnnop/level2/aclnn_cos.h"
#include "aclnnop/level2/aclnn_acos.h"
#include "ascend/ascend_kernel_without_config.h"
#include "ascend/ascend_runtime.h"

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
        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize,
                              ACL_MEM_MALLOC_HUGE_FIRST);
        }
        assert(ret == ACL_SUCCESS);
        ret = aclnnRelu(workspaceAddr, workspaceSize, executor,
                        context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);

	//ret = aclDestroyTensor(input);
        //assert(ret == ACL_SUCCESS);
	//ret = aclDestroyTensor(output);
        //assert(ret == ACL_SUCCESS);

        ret = aclrtSynchronizeStream(context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);


        return;
    }
};

class AbsAclnn : public ASCENDKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);

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
            aclnnAbsGetWorkspaceSize(input, output, &workspaceSize, &executor);
        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize,
                              ACL_MEM_MALLOC_HUGE_FIRST);
        }
        assert(ret == ACL_SUCCESS);
        ret = aclnnAbs(workspaceAddr, workspaceSize, executor,
                        context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);

	//ret = aclDestroyTensor(input);
        //assert(ret == ACL_SUCCESS);
	//ret = aclDestroyTensor(output);
        //assert(ret == ACL_SUCCESS);

        ret = aclrtSynchronizeStream(context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);

        return;
    }
};

class SigmoidAclnn : public ASCENDKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);

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
            aclnnSigmoidGetWorkspaceSize(input, output, &workspaceSize, &executor);
        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize,
                              ACL_MEM_MALLOC_HUGE_FIRST);
        }
        assert(ret == ACL_SUCCESS);
        ret = aclnnSigmoid(workspaceAddr, workspaceSize, executor,
                        context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);

	//ret = aclDestroyTensor(input);
        //assert(ret == ACL_SUCCESS);
	//ret = aclDestroyTensor(output);
        //assert(ret == ACL_SUCCESS);

        ret = aclrtSynchronizeStream(context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);

        return;
    }
};

class HardswishAclnn : public ASCENDKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryObj>(_op);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);

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
            aclnnHardswishGetWorkspaceSize(input, output, &workspaceSize, &executor);
        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize,
                              ACL_MEM_MALLOC_HUGE_FIRST);
        }
        assert(ret == ACL_SUCCESS);
        ret = aclnnHardswish(workspaceAddr, workspaceSize, executor,
                        context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);

	//ret = aclDestroyTensor(input);
        //assert(ret == ACL_SUCCESS);
	//ret = aclDestroyTensor(output);
        //assert(ret == ACL_SUCCESS);

        ret = aclrtSynchronizeStream(context->ASCENDHandle());
        assert(ret == ACL_SUCCESS);

        return;
    }
};


//class TanhAclnn : public ASCENDKernelWithoutConfig {
//    void compute(const Operator &_op,
//                 const RuntimeObj *_context) const override {
//        auto op = as<UnaryObj>(_op);
//        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);
//
//        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
//        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
//
//        auto a = op->getInputs(0)->getDims();
//        std::vector<int64_t> aDim(a.size(), 1);
//        for (size_t i = 0; i < a.size(); ++i) {
//            aDim[i] = int64_t(a[i]);
//        }
//        auto aS = op->getInputs(0)->getStride();
//        std::vector<int64_t> aStride(aS.size(), 1);
//        for (size_t i = 0; i < aS.size(); ++i) {
//            aStride[i] = int64_t(aS[i]);
//        }
//        auto c = op->getInputs(0)->getDims();
//        std::vector<int64_t> cDim(c.size(), 1);
//        for (size_t i = 0; i < c.size(); ++i) {
//            cDim[i] = int64_t(c[i]);
//        }
//        auto cS = op->getInputs(0)->getStride();
//        std::vector<int64_t> cStride(cS.size(), 1);
//        for (size_t i = 0; i < cS.size(); ++i) {
//            cStride[i] = int64_t(cS[i]);
//        }
//
//        auto input = aclCreateTensor(
//            aDim.data(), aDim.size(), ACL_FLOAT, aStride.data(), 0,
//            aclFormat::ACL_FORMAT_ND, aDim.data(), aDim.size(), aData);
//        auto output = aclCreateTensor(
//            cDim.data(), cDim.size(), ACL_FLOAT, cStride.data(), 0,
//            aclFormat::ACL_FORMAT_ND, cDim.data(), cDim.size(), cData);
//
//        uint64_t workspaceSize = 0;
//        aclOpExecutor *executor;
//
//        auto ret =
//            aclnnTanhGetWorkspaceSize(input, output, &workspaceSize, &executor);
//        void *workspaceAddr = nullptr;
//        if (workspaceSize > 0) {
//            ret = aclrtMalloc(&workspaceAddr, workspaceSize,
//                              ACL_MEM_MALLOC_HUGE_FIRST);
//        }
//        assert(ret == ACL_SUCCESS);
//        ret = aclnnTanh(workspaceAddr, workspaceSize, executor,
//                        context->ASCENDHandle());
//        assert(ret == ACL_SUCCESS);
//
//	//ret = aclDestroyTensor(input);
//        //assert(ret == ACL_SUCCESS);
//	//ret = aclDestroyTensor(output);
//        //assert(ret == ACL_SUCCESS);
//
//        ret = aclrtSynchronizeStream(context->ASCENDHandle());
//        assert(ret == ACL_SUCCESS);
//
//        return;
//    }
//};

#define DEFINE_UNARY_Aclnn(prefix)                                                 \
	class prefix##Aclnn : public ASCENDKernelWithoutConfig {                   \
	    void compute(const Operator &_op,                                      \
	                 const RuntimeObj *_context) const override {              \
	        auto op = as<UnaryObj>(_op);                                       \
	        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);   \
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
	        auto ret = aclnn##prefix##GetWorkspaceSize(input, output, &workspaceSize, &executor);  \
	        void *workspaceAddr = nullptr;                                     \
	        if (workspaceSize > 0) {                                           \
	            ret = aclrtMalloc(&workspaceAddr, workspaceSize,               \
	                              ACL_MEM_MALLOC_HUGE_FIRST);                  \
	        }                                                                  \
	        assert(ret == ACL_SUCCESS);                                        \
	        ret = aclnn##prefix(workspaceAddr, workspaceSize, executor,        \
	                        context->ASCENDHandle());                          \
	        assert(ret == ACL_SUCCESS);                                        \
	        ret = aclrtSynchronizeStream(context->ASCENDHandle());             \
	        assert(ret == ACL_SUCCESS);                                        \
	                                                                           \
	        return;                                                            \
	    }                                                                      \
	};

DEFINE_UNARY_Aclnn(Gelu)
DEFINE_UNARY_Aclnn(Tanh)
DEFINE_UNARY_Aclnn(Sin)
DEFINE_UNARY_Aclnn(Cos)
//DEFINE_UNARY_Aclnn(ACos)
//DEFINE_UNARY_Aclnn(Tan)

REGISTER_KERNEL(Device::ASCEND, OpType::Relu, DataType::Float32, ReluAclnn,
                "relu_ASCEND_float");
REGISTER_KERNEL(Device::ASCEND, OpType::Abs, DataType::Float32, AbsAclnn,
                "abs_ASCEND_float");
REGISTER_KERNEL(Device::ASCEND, OpType::Sigmoid, DataType::Float32, SigmoidAclnn,
                "sigmoid_ASCEND_float");
REGISTER_KERNEL(Device::ASCEND, OpType::HardSwish, DataType::Float32, HardswishAclnn,
                "hardswish_ASCEND_float");
REGISTER_KERNEL(Device::ASCEND, OpType::Tanh, DataType::Float32, TanhAclnn,
                "tanh_ASCEND_float");
REGISTER_KERNEL(Device::ASCEND, OpType::Gelu, DataType::Float32, GeluAclnn,
                "gelu_ASCEND_float");
REGISTER_KERNEL(Device::ASCEND, OpType::Sin, DataType::Float32, SinAclnn,
                "sin_ASCEND_float");
REGISTER_KERNEL(Device::ASCEND, OpType::Cos, DataType::Float32, CosAclnn,
                "cos_ASCEND_float");
//REGISTER_KERNEL(Device::ASCEND, OpType::ACos, DataType::Float32, ACosAclnn,
//                "acos_ASCEND_float");
//REGISTER_KERNEL(Device::ASCEND, OpType::Tan, DataType::Float32, TanAclnn,
//                "tan_ASCEND_float");
}; // namespace infini

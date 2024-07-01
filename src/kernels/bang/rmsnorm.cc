#include "operators/rms_norm.h"
#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"
#include "bang/bang_rmsnorm.h"
namespace infini {
class RMSNormBang : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<RMSNormObj>(_op);
        auto input = op->getInputs(0);
        auto weight = op->getInputs(1);
        auto output = op->getOutput();
        void *const mlu_src = input->getRawDataPtr<void *>();
        void *const mlu_weight = weight->getRawDataPtr<void *>();
        void *const mlu_destination = output->getRawDataPtr<void *>();
        const auto &inputShape = input->getDims();
        int nDims = input->getDims().size();
        
        int dimsize = inputShape[nDims - 1];
        int othersize = input->size() / dimsize;
        
        IT_ASSERT(dimsize == (int)weight->size());

        auto context = dynamic_cast<const BangRuntimeObj *>(_context);
        float eps = 1e-5;
        if (op->getOpType() == OpType::RMSNorm)
            rmsNormKernel(context->cnnlHandle(), (float *)mlu_destination, (float *)mlu_src, (float *)mlu_weight, othersize, dimsize, eps);

        else
            IT_TODO_HALT();
    }
};

REGISTER_KERNEL(Device::BANG, OpType::RMSNorm, RMSNormBang, "RMSNorm_BANG");
}; // namespace infini

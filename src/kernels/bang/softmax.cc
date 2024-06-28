#include "operators/softmax.h"
#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"
#include "bang/bang_softmax.h"
namespace infini {
class SoftmaxBang : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<BangSoftmaxObj>(_op);
        void *const mlu_src = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const mlu_destination = (op->getOutput()->getRawDataPtr<void *>());

        auto context = dynamic_cast<const BangRuntimeObj *>(_context);
        auto shape = op->getInputs(0)->getDims();
        int nDim = shape.size();
        int axis = op->getAxis();
        int stride = 1;
        int dimsize = shape[axis];
        int num = 1;
        int othersize = 1;
        int frontsize = 1;

        for (int s = nDim - 1; s >= 0; s--) {
            num *= shape[s];
            if (s > axis) {
                stride *= shape[s];
            }
            if (s < axis) {
                frontsize *= shape[s];
            }
            if (s != axis) {
                othersize *= shape[s];
            }
        }
        if (op->getOpType() == OpType::BangSoftmax)
            softmaxKernel(context->cnnlHandle(), (float *)mlu_destination, (float *)mlu_src, othersize, dimsize, frontsize, stride, axis, nDim);
            
        else
            IT_TODO_HALT();
    }
};

REGISTER_KERNEL(Device::BANG, OpType::BangSoftmax, SoftmaxBang, "Softmax_BANG");
}; // namespace infini


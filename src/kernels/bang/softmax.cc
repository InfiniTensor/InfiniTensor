#include "operators/softmax.h"
#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"
#include "bang/bang_softmax.h"
namespace infini {
class SoftmaxBang : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        softmax_kernel(_context, _op);
    }
};

REGISTER_KERNEL(Device::BANG, OpType::BangSoftmax, SoftmaxBang, "Softmax_BANG");
}; // namespace infini

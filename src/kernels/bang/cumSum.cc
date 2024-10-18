#include "operators/unary.h"
#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"

namespace infini {
class CumSumCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {

    }
};

REGISTER_KERNEL(Device::BANG, OpType::CumSum, CumSumCnnl, "CumSum_cnnl_BANG");

}; // namespace infini

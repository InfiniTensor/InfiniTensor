#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"
#include "bang/bang_unary_list.h"
#include "operators/unary.h"

namespace infini {
class UnaryKernel : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        bang_unary_kernel(_context, _op);
    }
};

REGISTER_KERNEL(Device::BANG, OpType::UnaryKernel, DataType::Float32,
                UnaryKernel, "Unary_BANG_Float32");

}; // namespace infini

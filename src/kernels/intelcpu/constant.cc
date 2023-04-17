#include "operators/constant.h"
#include "intelcpu/mkl_kernel_without_config.h"

namespace infini {
class ConstantKernel : public MklKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        ;
    }
};
REGISTER_KERNEL(Device::INTELCPU, OpType::Constant, DataType::Int32,
                ConstantKernel, "Constant_Mkl_Int32");
REGISTER_KERNEL(Device::INTELCPU, OpType::Constant, DataType::Float32,
                ConstantKernel, "Constant_Mkl_Float32");
} // namespace infini

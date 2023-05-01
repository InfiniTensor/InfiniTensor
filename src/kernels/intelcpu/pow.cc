#include "core/kernel.h"
#include "intelcpu/mkl_kernel_without_config.h"
#include "intelcpu/mkl_runtime.h"
#include "operators/element_wise.h"
#include <math.h>

namespace infini {
class MklPow : public MklKernelWithoutConfig {
    // TODO: not need to copy memory??
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<PowObj>(_op);
        auto in0Data = op->getInputs(0)->getRawDataPtr<float *>();
        auto in1Data = op->getInputs(1)->getRawDataPtr<float *>();
        auto outData = op->getOutput(0)->getRawDataPtr<float *>();
        auto size = op->getInputs(0)->size();

#pragma omp parallel for
        for (size_t i = 0; i < size; ++i) {
            outData[i] = pow(in0Data[i], in1Data[i]);
        }
    }
};

REGISTER_KERNEL(Device::INTELCPU, OpType::Pow, DataType::Float32, MklPow,
                "Pow_Mkl_Float32");

}; // namespace infini

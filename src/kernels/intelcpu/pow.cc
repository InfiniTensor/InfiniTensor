#include "core/kernel.h"
#include "intelcpu/mkl_kernel_without_config.h"
#include "intelcpu/mkl_runtime.h"
#include "operators/element_wise.h"
#include <CL/sycl.hpp>
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
        int size = op->getInputs(0)->size();

        // cpu_selector using openCL as backend;and host_selector bypasses the
        // OnenCL backend and runs directly on CPU hardware
        sycl::queue q(sycl::cpu_selector{});
        auto in0Device = sycl::malloc_device<float>(size, q);
        auto in1Device = sycl::malloc_device<float>(size, q);
        auto outDevice = sycl::malloc_device<float>(size, q);
        q.memcpy(in0Device, in0Data, size * sizeof(float));
        q.wait();
        q.memcpy(in1Device, in1Data, size * sizeof(float));
        q.wait();

        q.parallel_for(sycl::range<1>(size), [=](sycl::id<1> i) {
             outDevice[i] = pow(in0Device[i], in1Device[i]);
         }).wait();
        q.memcpy(outData, outDevice, size * sizeof(float));
        q.wait();
        sycl::free(in0Device, q);
        sycl::free(in1Device, q);
        sycl::free(outDevice, q);
    }
};
REGISTER_KERNEL(Device::INTELCPU, OpType::Pow, DataType::Float32, MklPow,
                "Pow_Mkl_Float32");

}; // namespace infini

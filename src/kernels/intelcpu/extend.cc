#include "operators/extend.h"
#include "core/kernel.h"
#include "intelcpu/mkl_kernel_without_config.h"
#include "intelcpu/mkl_runtime.h"
#include <CL/sycl.hpp>
#include <math.h>

namespace infini {
class MklExtend : public MklKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ExtendObj>(_op);
        auto inData = op->getInputs(0)->getRawDataPtr<float *>();
        auto outData = op->getOutput(0)->getRawDataPtr<float *>();
        int iSize = op->getInputs(0)->size();
        int oSize = op->getOutput(0)->size();

        sycl::queue q(sycl::cpu_selector{});
        auto inDevice = sycl::malloc_device<float>(iSize, q);
        auto outDevice = sycl::malloc_device<float>(oSize, q);

        q.memcpy(inDevice, inData, iSize * sizeof(float));
        q.wait();

        int blockSize = 1;
        auto iDim = op->getInputs(0)->getDims();
        for (size_t i = iDim.size() - 1;
             i >= (size_t)op->getDim() && i != (size_t)-1; --i)
            blockSize *= iDim[i];
        auto blockSizeOuter = (op->getNum() + 1) * blockSize;

        q.parallel_for(sycl::range<1>(oSize), [=](sycl::id<1> index) {
             auto iIdx = index % blockSize + index / blockSizeOuter * blockSize;
             outDevice[index] = inDevice[iIdx];
         }).wait();

        q.memcpy(outData, outDevice, oSize * sizeof(float));
        q.wait();
        sycl::free(inDevice, q);
        sycl::free(outDevice, q);
    }
};
REGISTER_KERNEL(Device::INTELCPU, OpType::Extend, DataType::Float32, MklExtend,
                "Extend_Mkl_Float32");
}; // namespace infini

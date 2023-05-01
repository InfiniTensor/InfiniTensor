#include "operators/extend.h"
#include "core/kernel.h"
#include "intelcpu/mkl_kernel_without_config.h"
#include "intelcpu/mkl_runtime.h"
#include <math.h>

namespace infini {
class MklExtend : public MklKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ExtendObj>(_op);
        auto input = op->getInputs(0);
        auto output = op->getOutput();
        auto inData = input->getRawDataPtr<float *>();
        auto outData = output->getRawDataPtr<float *>();
        auto oSize = output->size();

        int blockSize = 1;
        auto iDim = input->getDims();
        auto dim = op->getDim();
        auto num = op->getNum();
        for (size_t i = iDim.size() - 1; i >= (size_t)dim && i != (size_t)-1;
             --i)
            blockSize *= iDim[i];
        auto blockSizeOuter = (num + 1) * blockSize;

#pragma omp parallel for
        for (size_t oIdx = 0; oIdx < oSize; ++oIdx) {
            auto iIdx = oIdx % blockSize + oIdx / blockSizeOuter * blockSize;
            outData[oIdx] = inData[iIdx];
        }
    }
};
REGISTER_KERNEL(Device::INTELCPU, OpType::Extend, DataType::Float32, MklExtend,
                "Extend_Mkl_Float32");
}; // namespace infini

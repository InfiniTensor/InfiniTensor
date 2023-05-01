#include "operators/gather.h"
#include "core/kernel.h"
#include "intelcpu/mkl_kernel_without_config.h"
#include "intelcpu/mkl_runtime.h"
#include <math.h>

namespace infini {
class MklGather : public MklKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<GatherObj>(_op);
        auto input = op->getInputs(0);
        auto index = op->getInputs(1);
        auto output = op->getOutput();
        auto inData = input->getRawDataPtr<float *>();
        auto indexData = index->getRawDataPtr<uint32_t *>();
        auto outData = output->getRawDataPtr<float *>();
        auto oSize = output->size();

        int inNDim = input->getDims().size();
        int oNDim = output->getDims().size();
        int idxNDim = index->getDims().size();
        int axis = op->getAxis();

        int outDim[4] = {0};
        int idxDim[4] = {0};
        int idxStride[4] = {0};
        int inStride[4] = {0};
        for (int i = 0; i < oNDim; ++i)
            outDim[i] = output->getDims()[i];
        for (int i = 0; i < idxNDim; ++i) {
            idxDim[i] = index->getDims()[i];
            idxStride[i] = index->getStride()[i];
        }
        for (int i = 0; i < inNDim; ++i) {
            inStride[i] = input->getStride()[i];
        }

#pragma omp parallel for
        for (size_t index = 0; index < oSize; ++index) {
            int offset = 0;
            int gOffset = index;
            for (int i = inNDim - 1, k = oNDim - 1; i >= 0; --i) {
                int idx = 0;
                if (i == axis) {
                    int idxOffset = 0;
                    for (int j = idxNDim - 1; j >= 0; --j) {
                        int p = gOffset % idxDim[j];
                        gOffset = gOffset / idxDim[j];
                        idxOffset += p * idxStride[j];
                    }

                    idx = indexData[idxOffset];
                    k = k - idxNDim;

                } else {
                    idx = gOffset % outDim[k];
                    gOffset = gOffset / outDim[k];
                    --k;
                }
                offset += idx * inStride[i];
            }

            outData[index] = inData[offset];
        }
    }
};

REGISTER_KERNEL(Device::INTELCPU, OpType::Gather, DataType::Float32, MklGather,
                "Gather_Mkl_Float32");
}; // namespace infini

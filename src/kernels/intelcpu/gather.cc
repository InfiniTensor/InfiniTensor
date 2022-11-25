#include "operators/gather.h"
#include "core/kernel.h"
#include "intelcpu/mkl_kernel_without_config.h"
#include "intelcpu/mkl_runtime.h"
#include <CL/sycl.hpp>
#include <math.h>

namespace infini {
class MklGather : public MklKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<GatherObj>(_op);
        auto in = op->getInputs(0);
        auto index = op->getInputs(1);
        auto out = op->getOutput();
        int iSize = in->size();
        int oSize = out->size();
        int idxSize = index->size();

        int inNDim = in->getDims().size();
        int oNDim = out->getDims().size();
        int idxNDim = index->getDims().size();
        int axis = op->getAxis();

        int outDim[4] = {0};
        int idxDim[4] = {0};
        int idxStride[4] = {0};
        int inStride[4] = {0};
        for (int i = 0; i < oNDim; ++i)
            outDim[i] = out->getDims()[i];
        for (int i = 0; i < idxNDim; ++i) {
            idxDim[i] = index->getDims()[i];
            idxStride[i] = index->getStride()[i];
        }
        for (int i = 0; i < inNDim; ++i) {
            inStride[i] = in->getStride()[i];
        }

        sycl::queue q(sycl::cpu_selector{});
        auto inDevice = sycl::malloc_device<float>(iSize, q);
        auto indexDevice = sycl::malloc_device<uint32_t>(idxSize, q);
        auto outDevice = sycl::malloc_device<float>(oSize, q);

        q.memcpy(inDevice, in->getRawDataPtr<float *>(), iSize * sizeof(float));
        q.memcpy(indexDevice, index->getRawDataPtr<uint32_t *>(),
                 idxSize * sizeof(uint32_t));
        q.wait();

        q.parallel_for(sycl::range<1>(oSize), [=](sycl::id<1> index) {
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

                     idx = indexDevice[idxOffset];
                     k = k - idxNDim;

                 } else {
                     idx = gOffset % outDim[k];
                     gOffset = gOffset / outDim[k];
                     --k;
                 }
                 offset += idx * inStride[i];
             }

             outDevice[index] = inDevice[offset];
         }).wait();

        q.memcpy(out->getRawDataPtr<float *>(), outDevice,
                 oSize * sizeof(float));
        q.wait();
        sycl::free(inDevice, q);
        sycl::free(outDevice, q);
        sycl::free(indexDevice, q);
    }
};
REGISTER_KERNEL(Device::INTELCPU, OpType::Gather, DataType::Float32, MklGather,
                "Gather_Mkl_Float32");
}; // namespace infini

#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_pad_slice.h"
#include "operators/pad.h"
#include "operators/slice.h"
namespace infini {
class PadSliceCudaCompute {
  public:
    void do_compute(Tensor partTensor, Tensor wholeTensor, const Shape &begNos,
                    bool isPad) const {
        int nDims = partTensor->getRank();
        IT_ASSERT(MAX_DIM >= nDims);
        TransMetaData metadata;
        for (int i = 0; i < nDims; i++) {
            metadata.begNum[i] = begNos[i];
            metadata.wholeNDim[i] = wholeTensor->getDims()[i];
            metadata.partNDim[i] = partTensor->getDims()[i];
            metadata.partStride[i] = partTensor->getStride()[i];
        }
        metadata.DType = partTensor->getDType().getIndex();
        pad_slice_kernel(partTensor->getRawDataPtr<void *>(),
                         wholeTensor->getRawDataPtr<void *>(), metadata, nDims,
                         wholeTensor->size(), isPad);
    }
};

class PadCuda : private PadSliceCudaCompute, public CudaKernelWithoutConfig {
    void compute(const Operator &op,
                 const RuntimeObj *_context) const override {
        do_compute(op->getInputs(0), op->getOutput(), as<PadObj>(op)->getPads(),
                   true);
    }
};

class SliceCuda : private PadSliceCudaCompute, public CudaKernelWithoutConfig {
    void compute(const Operator &op,
                 const RuntimeObj *_context) const override {
        do_compute(op->getOutput(), op->getInputs(0),
                   as<SliceObj>(op)->getStarts(), false);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Slice, SliceCuda, "Slice__CUDA");

REGISTER_KERNEL(Device::CUDA, OpType::Pad, PadCuda, "Pad__CUDA");

} // namespace infini

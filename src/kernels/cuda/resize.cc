#include "operators/resize.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/resize.cuh"
namespace infini {
class ResizeCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ResizeObj>(_op);
        auto in = op->getInputs(0);
        auto out = op->getOutputs()[0];

        int nDims = in->getDims().size();
        if (nDims > 4)
            IT_TODO_HALT();

        MetaData metaData;
        memset(&metaData, 0, sizeof(metaData));
        metaData.nDims = nDims;
        for (int i = 0; i < nDims; ++i) {
            metaData.inDims[i] = in->getDims()[i];
            metaData.oDims[i] = out->getDims()[i];
            metaData.inStride[i] = in->getStride()[i];
            metaData.scale[i] = op->getScale(i);
            metaData.roiS[i] = op->getRoi(i);
            metaData.roiE[i] = op->getRoi(i + nDims);
        }

        switch (op->getMode()) {
        case ResizeObj::ECoeffMode::nearest:
            resize_kernel_nearest(in->getRawDataPtr<float *>(),
                                  out->getRawDataPtr<float *>(), metaData,
                                  out->size(), op->getCoordinateTransMode(),
                                  op->getNearestMode());
            break;
        case ResizeObj::ECoeffMode::linear:
            resize_kernel_linear(in->getRawDataPtr<float *>(),
                                 out->getRawDataPtr<float *>(), metaData,
                                 out->size(), op->getCoordinateTransMode());
            break;
        case ResizeObj::ECoeffMode::cubic:
            resize_kernel_cubic(in->getRawDataPtr<float *>(),
                                out->getRawDataPtr<float *>(), metaData,
                                out->size(), op->getCoordinateTransMode());
            break;
        default:
            IT_TODO_HALT();
        }
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Resize, DataType::Float32, ResizeCuda,
                "Resize_CUDA_Float32");

} // namespace infini

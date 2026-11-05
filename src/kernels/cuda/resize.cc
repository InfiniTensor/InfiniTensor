#include "operators/resize.h"
#include "cuda/cuda_kernel_without_config.h"
#include "cuda/resize.cuh"
namespace infini {
class ResizeCuda : public CudaKernelWithoutConfig {
    // Instantiated separately for half and float. Dispatch is based only on
    // Resize's data input, not on scales/ROI dtype or a global precision flag.
    template <typename T>
    static void computeTyped(const Ref<ResizeObj> &op, T *input, T *output,
                             const MetaData &metaData, size_t num) {
        switch (op->getMode()) {
        case ResizeObj::ECoeffMode::nearest:
            resize_kernel_nearest(input, output, metaData, num,
                                  op->getCoordinateTransMode(),
                                  op->getNearestMode());
            break;
        case ResizeObj::ECoeffMode::linear:
            resize_kernel_linear(input, output, metaData, num,
                                 op->getCoordinateTransMode());
            break;
        case ResizeObj::ECoeffMode::cubic:
            resize_kernel_cubic(input, output, metaData, num,
                                op->getCoordinateTransMode());
            break;
        default:
            IT_TODO_HALT();
        }
    }

    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ResizeObj>(_op);
        auto in = op->getInputs(0);
        auto out = op->getOutputs()[0];
        const auto inputDType = in->getDType();
        IT_ASSERT(inputDType == DataType::Float16 ||
                      inputDType == DataType::Float32,
                  "CUDA Resize supports Float16/Float32 data inputs only");
        IT_ASSERT(out->getDType() == inputDType,
                  "Resize output dtype must match its data input");
        void *inData = in->getRawDataPtr<void *>();
        void *outData = out->getRawDataPtr<void *>();

        int nDims = in->getRank();
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

        // Coordinates, scales and ROI intentionally remain float. No tensor
        // cast or temporary FP32 data buffer is introduced by this dispatch.
        if (inputDType == DataType::Float16) {
            computeTyped<half>(op, static_cast<half *>(inData),
                               static_cast<half *>(outData), metaData,
                               out->size());
        } else {
            computeTyped<float>(op, static_cast<float *>(inData),
                                static_cast<float *>(outData), metaData,
                                out->size());
        }
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Resize, ResizeCuda, "Resize_CUDA");

} // namespace infini

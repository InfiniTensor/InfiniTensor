#include "operators/resize.h"
#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"
#include <iostream>

namespace infini {
class ResizeCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ResizeObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto nDims = op->getInputs(0)->getRank();
        if (nDims != 4) {
            IT_TODO_HALT();
        }
        auto aDim = op->getInputs(0)->getDims();
        auto cDim = op->getOutput()->getDims();
        std::vector<int> aTransDim = {aDim[0], aDim[2], aDim[3], aDim[1]};
        std::vector<int> cTransDim = {cDim[0], cDim[2], cDim[3], cDim[1]};

        cnnlTensorDescriptor_t aDesc, cDesc, aTransDesc, cTransDesc;
        // input
        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            aDesc, CNNL_LAYOUT_NCHW, cnnlDataTypeConvert(op->getDType()),
            aDim.size(), aDim.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&aTransDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            aTransDesc, CNNL_LAYOUT_NHWC, cnnlDataTypeConvert(op->getDType()),
            aTransDim.size(), aTransDim.data()));
        // output
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            cDesc, CNNL_LAYOUT_NCHW, cnnlDataTypeConvert(op->getDType()),
            cDim.size(), cDim.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&cTransDesc));
        checkCnnlError(cnnlSetTensorDescriptor(
            cTransDesc, CNNL_LAYOUT_NHWC, cnnlDataTypeConvert(op->getDType()),
            cTransDim.size(), cTransDim.data()));

        // transpose
        BangPtr aTransData = context->getWorkspace(
            cnnlGetTensorElementNum(aTransDesc) * op->getDType().getSize());
        BangPtr cTransData = context->getWorkspace(
            cnnlGetTensorElementNum(cTransDesc) * op->getDType().getSize());

        int permuteIn[4] = {0, 2, 3, 1};
        cnnlTransposeDescriptor_t inDesc;
        checkCnnlError(cnnlCreateTransposeDescriptor(&inDesc));
        checkCnnlError(cnnlSetTransposeDescriptor(inDesc, 4, permuteIn));
        size_t wsSizeIn;
        cnnlGetTransposeWorkspaceSize(context->cnnlHandle(), aDesc, inDesc,
                                      &wsSizeIn);
        BangPtr wsDataIn = context->getWorkspace(wsSizeIn);

        checkCnnlError(cnnlTranspose_v2(context->cnnlHandle(), inDesc, aDesc,
                                        aData, aTransDesc, aTransData, wsDataIn,
                                        wsSizeIn));

        cnnlTensorDescriptor_t boxesDesc, boxesIndexDesc;
        checkCnnlError(cnnlCreateTensorDescriptor(&boxesDesc));
        auto nBatch = aDim[0];
        std::vector<int> boxesDim = {nBatch, 4};
        checkCnnlError(cnnlSetTensorDescriptor(
            boxesDesc, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(op->getDType()),
            boxesDim.size(), boxesDim.data()));

        checkCnnlError(cnnlCreateTensorDescriptor(&boxesIndexDesc));
        std::vector<int> boxesIndexDim = {nBatch};
        checkCnnlError(cnnlSetTensorDescriptor(
            boxesIndexDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32,
            boxesIndexDim.size(), boxesIndexDim.data()));
        std::vector<int32_t> boxesIndex(nBatch);
        std::iota(boxesIndex.begin(), boxesIndex.end(), 0);
        BangPtr boxesIndexData =
            context->getWorkspace(nBatch * sizeof(int32_t));
        context->copyBlobFromCPU(boxesIndexData, boxesIndex.data(),
                                 nBatch * sizeof(int32_t));

        cnnlCropAndResizeMode_t mode;
        auto coefMode = op->getMode();
        if (coefMode == ResizeObj::ECoeffMode::nearest) {
            mode = CNNL_CROP_AND_RESIZE_NEAREST;
        } else if (coefMode == ResizeObj::ECoeffMode::linear) {
            mode = CNNL_CROP_AND_RESIZE_BILINEAR;
        } else {
            IT_TODO_HALT();
        }

        std::vector<float> box;
        auto transMode = op->getCoordinateTransMode();
        if (transMode ==
            enum_to_underlying(
                ResizeObj::ECoordinateTransMode::tfCropAndResize)) {
            box = {op->getRoi(2), op->getRoi(3), op->getRoi(6), op->getRoi(7)};
        } else {
            box = {0, 0, 1.0, 1.0};
        }

        BangPtr boxesData =
            context->getWorkspace(nBatch * box.size() * sizeof(float));
        for (auto i = 0; i < nBatch; i++) {
            context->copyBlobFromCPU(boxesData + i * box.size() * sizeof(float),
                                     box.data(), box.size() * sizeof(float));
        }

        checkCnnlError(cnnlCropAndResize(
            context->cnnlHandle(), aTransDesc, aTransData, boxesDesc, boxesData,
            boxesIndexDesc, boxesIndexData, mode, 0.0, cTransDesc, cTransData));

        // transpose
        int permuteOut[4] = {0, 3, 1, 2};
        cnnlTransposeDescriptor_t outDesc;
        checkCnnlError(cnnlCreateTransposeDescriptor(&outDesc));
        checkCnnlError(cnnlSetTransposeDescriptor(outDesc, 4, permuteOut));
        size_t wsSizeOut;
        cnnlGetTransposeWorkspaceSize(context->cnnlHandle(), cTransDesc,
                                      outDesc, &wsSizeOut);
        BangPtr wsDataOut = context->getWorkspace(wsSizeOut);

        checkCnnlError(cnnlTranspose_v2(context->cnnlHandle(), outDesc,
                                        cTransDesc, cTransData, cDesc, cData,
                                        wsDataOut, wsSizeOut));

        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(aTransDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cTransDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(boxesDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(boxesIndexDesc));
        checkCnnlError(cnnlDestroyTransposeDescriptor(inDesc));
        checkCnnlError(cnnlDestroyTransposeDescriptor(outDesc));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::Resize, ResizeCnnl, "Resize_cnnl_BANG");
}; // namespace infini

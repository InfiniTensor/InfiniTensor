#pragma GCC diagnostic ignored "-Wunused-variable"
#include "operators/where.h"
#include "kunlun/kunlun_kernel_without_config.h"
#include "kunlun/kunlun_runtime.h"
#include "utils/operator_utils.h"

namespace infini {

class WhereXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<WhereObj>(_op);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData =
            (op->getInputs(0)->getRawDataPtr<void *>()); // inputX
        void *const bData =
            (op->getInputs(1)->getRawDataPtr<void *>()); // inputY
        void *const cData =
            (op->getInputs(2)->getRawDataPtr<void *>()); // condition
        void *const dData =
            (op->getOutput()->getRawDataPtr<void *>()); // output

        auto aDim = op->getInputs(0)->getDims(); // dimX
        auto bDim = op->getInputs(1)->getDims(); // dimY
        auto cDim = op->getInputs(2)->getDims(); // dimCondition
        auto dDim = op->getOutput()->getDims();  //  dimOutput

        auto dtype = op->getDType();

        if (aDim != bDim) {
            // Infer broadcast for X and Y
            Shape XYDim = infer_broadcast(aDim, bDim);
            int XYSize = std::accumulate(XYDim.begin(), XYDim.end(), 1,
                                         std::multiplies<int>());
            // Align rank for XYDim and aDim or bDim
            broadcastShape(XYDim, aDim);
            broadcastShape(XYDim, bDim);
            // Get workspace
            void *wkspace = context->getWorkspace(XYSize * dtype.getSize());
            // Broadcast X Y
            if (dtype == DataType::Float32) {
                checkKUNLUNError(xdnn::broadcast<float>(
                    context->KUNLUNHandle(),
                    (float *)(XYDim == aDim ? bData : aData), (float *)wkspace,
                    (XYDim == aDim ? bDim : aDim), XYDim));
            } else if (dtype == DataType::Float16) {
                checkKUNLUNError(xdnn::broadcast<float16>(
                    context->KUNLUNHandle(),
                    (float16 *)(XYDim == aDim ? bData : aData),
                    (float16 *)wkspace, (XYDim == aDim ? bDim : aDim), XYDim));
            } else if (dtype == DataType::Int32) {
                checkKUNLUNError(xdnn::broadcast<int>(
                    context->KUNLUNHandle(),
                    (int *)(XYDim == aDim ? bData : aData), (int *)wkspace,
                    (XYDim == aDim ? bDim : aDim), XYDim));
            } else if (dtype == DataType::Int8) {
                checkKUNLUNError(xdnn::broadcast<int8_t>(
                    context->KUNLUNHandle(),
                    (int8_t *)(XYDim == aDim ? bData : aData),
                    (int8_t *)wkspace, (XYDim == aDim ? bDim : aDim), XYDim));
            } else if (dtype == DataType::Int64) {
                checkKUNLUNError(xdnn::broadcast<int64_t>(
                    context->KUNLUNHandle(),
                    (int64_t *)(XYDim == aDim ? bData : aData),
                    (int64_t *)wkspace, (XYDim == aDim ? bDim : aDim), XYDim));
            } else if (dtype == DataType::Int16) {
                checkKUNLUNError(xdnn::broadcast<int16_t>(
                    context->KUNLUNHandle(),
                    (int16_t *)(XYDim == aDim ? bData : aData),
                    (int16_t *)wkspace, (XYDim == aDim ? bDim : aDim), XYDim));
            } else {
                IT_ASSERT(false,
                          "unsupported data type " + op->getDType().toString());
            }

            // Align Rank
            broadcastShape(dDim, XYDim);
            broadcastShape(dDim, XYDim);
            // Where
            void *XData = XYDim == aDim ? aData : wkspace;
            void *YData = XYDim == bDim ? bData : wkspace;
            if (dtype == DataType::Float32) {
                checkKUNLUNError(xdnn::select<float>(
                    context->KUNLUNHandle(), (bool *)cData, (float *)XData,
                    (float *)YData, (float *)dData, cDim, XYDim));
            } else if (dtype == DataType::Float16) {
                checkKUNLUNError(xdnn::select<float16>(
                    context->KUNLUNHandle(), (bool *)cData, (float16 *)XData,
                    (float16 *)YData, (float16 *)dData, cDim, XYDim));
            } else if (dtype == DataType::Int32) {
                checkKUNLUNError(xdnn::select<int>(
                    context->KUNLUNHandle(), (bool *)cData, (int *)XData,
                    (int *)YData, (int *)dData, cDim, XYDim));
            } else if (dtype == DataType::Int8) {
                checkKUNLUNError(xdnn::select<int8_t>(
                    context->KUNLUNHandle(), (bool *)cData, (int8_t *)XData,
                    (int8_t *)YData, (int8_t *)dData, cDim, XYDim));
            } else if (dtype == DataType::Int64) {
                checkKUNLUNError(xdnn::select<int64_t>(
                    context->KUNLUNHandle(), (bool *)cData, (int64_t *)XData,
                    (int64_t *)YData, (int64_t *)dData, cDim, XYDim));
            } else {
                IT_ASSERT(false,
                          "unsupported data type " + op->getDType().toString());
            }
        } else {
            if (dtype == DataType::Float32) {
                checkKUNLUNError(xdnn::select<float>(
                    context->KUNLUNHandle(), (bool *)cData, (float *)aData,
                    (float *)bData, (float *)dData, cDim, aDim));
            } else if (dtype == DataType::Float16) {
                checkKUNLUNError(xdnn::select<float16>(
                    context->KUNLUNHandle(), (bool *)cData, (float16 *)aData,
                    (float16 *)bData, (float16 *)dData, cDim, aDim));
            } else if (dtype == DataType::Int32) {
                checkKUNLUNError(xdnn::select<int>(
                    context->KUNLUNHandle(), (bool *)cData, (int *)aData,
                    (int *)bData, (int *)dData, cDim, aDim));
            } else if (dtype == DataType::Int8) {
                checkKUNLUNError(xdnn::select<int8_t>(
                    context->KUNLUNHandle(), (bool *)cData, (int8_t *)aData,
                    (int8_t *)bData, (int8_t *)dData, cDim, aDim));
            } else if (dtype == DataType::Int64) {
                checkKUNLUNError(xdnn::select<int64_t>(
                    context->KUNLUNHandle(), (bool *)cData, (int64_t *)aData,
                    (int64_t *)bData, (int64_t *)dData, cDim, aDim));
            } else {
                IT_ASSERT(false,
                          "unsupported data type " + op->getDType().toString());
            }
        }

        return;
    }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::Where, WhereXdnn, "Where_xdnn_KUNLUN");
}; // namespace infini


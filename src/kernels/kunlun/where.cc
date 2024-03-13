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
        IT_ASSERT(op->getDType() == DataType::Float32);
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
            checkKUNLUNError(xdnn::broadcast<float>(
                context->KUNLUNHandle(),
                (float *)(XYDim == aDim ? bData : aData), (float *)wkspace,
                (XYDim == aDim ? bDim : aDim), XYDim));
            // Align Rank
            broadcastShape(dDim, XYDim);
            broadcastShape(dDim, XYDim);
            // Where
            void *XData = XYDim == aDim ? aData : wkspace;
            void *YData = XYDim == bDim ? bData : wkspace;
            checkKUNLUNError(xdnn::select<float>(
                context->KUNLUNHandle(), (bool *)cData, (float *)XData,
                (float *)YData, (float *)dData, cDim, XYDim));
        } else {
            checkKUNLUNError(xdnn::select<float>(
                context->KUNLUNHandle(), (bool *)cData, (float *)aData,
                (float *)bData, (float *)dData, cDim, aDim));
        }

        return;
    }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::Where, WhereXdnn, "Where_xdnn_KUNLUN");
}; // namespace infini

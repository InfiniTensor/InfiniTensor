#include "operators/resize.h"
#include "kunlun/kunlun_kernel_without_config.h"
#include "kunlun/kunlun_runtime.h"

namespace infini {
class ResizeXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ResizeObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        auto ndims = op->getInputs(0)->getRank();
        IT_ASSERT(ndims == 4);
        // Resize mode
        auto mode = op->getMode();

        auto inShape = op->getInputs(0)->getDims();
        auto outShape = op->getOutput()->getDims();

        void *inData = op->getInputs(0)->getRawDataPtr<void *>();
        void *outData = op->getOutput()->getRawDataPtr<void *>();

        auto n = inShape[0];
        auto c = inShape[1];
        auto xh = inShape[2];
        auto xw = inShape[3];
        auto yh = outShape[2];
        auto yw = outShape[3];

        auto coMode = op->getCoordinateTransMode();
        auto coModeXdnn = 1;
        switch (coMode) {
        case 2:
            coModeXdnn = 0;
            break;
        case 0:
            coModeXdnn = 1;
            break;
        case 3:
            coModeXdnn = 2;
            break;
        default:
            IT_TODO_HALT();
        }

        switch (mode) {
        case ResizeObj::ECoeffMode::linear: {
            IT_TODO_HALT();
        }
        case ResizeObj::ECoeffMode::nearest: {
            auto nearest_mode = op->getNearestMode();
            auto nearestModeXdnn = 1;
            switch (nearest_mode) {
            case 2:
                nearestModeXdnn = 1;
                break;
            case 1:
                nearestModeXdnn = 0;
                break;
            default:
                IT_TODO_HALT();
            }
            checkKUNLUNError(xdnn::nearest_resize2d(
                context->KUNLUNHandle(), (float *)inData, (float *)outData, n,
                c, xh, xw, yh, yw, coModeXdnn, nearestModeXdnn, true));
            break;
        }
        default:
            IT_TODO_HALT();
        }
        return;
    }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::Resize, ResizeXdnn,
                "Resize_xdnn_KUNLUN");

}; // namespace infini

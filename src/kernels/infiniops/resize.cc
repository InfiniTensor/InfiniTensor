#include "operators/resize.h"
#include "infiniops_common.h"
#include "infiniops_dispatch.h"

#include <base/resize2d.h>

namespace infini {
namespace {

class ResizeInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op, const RuntimeObj *context) const override {
        const auto op = as<ResizeObj>(_op);
        using Mode = ResizeObj::ECoeffMode;
        using Resize = ::infini::ops::Resize2d;
        IT_ASSERT(op->getInputs(0)->getRank() == 4, "Infini Resize requires NCHW rank four");
        IT_ASSERT(op->getMode() == Mode::nearest || op->getMode() == Mode::linear,
                  "Infini Resize supports nearest and linear interpolation");
        const auto coordinate = op->getCoordinateTransMode();
        IT_ASSERT(coordinate >= Resize::kHalfPixel && coordinate <= Resize::kAsymmetric,
                  "Infini Resize does not support this coordinate transformation");
        const auto mode = op->getMode() == Mode::nearest ? Resize::kNearest : Resize::kLinear;
        const auto rounding = mode == Resize::kNearest ? op->getNearestMode() : Resize::kPreferFloor;
        auto input = infiniops::toInfiniOpsTensor(op->getInputs(0), context);
        auto out = infiniops::toInfiniOpsTensor(op->getOutput(), context);
        auto handle = infiniops::makeInfiniOpsHandle(context);
        auto config = infiniops::makeInfiniOpsConfig<Resize>(context);
        const auto heightScale = op->isResizeBySizes() ? 0.0f : op->getScale(2);
        const auto widthScale = op->isResizeBySizes() ? 0.0f : op->getScale(3);
        infiniops::dispatch::callResize2d(handle, config, input, mode, coordinate,
                                         rounding, heightScale, widthScale, out);
    }
};

} // namespace

REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Resize, ResizeInfiniOps,
                "Resize_InfiniOps");

} // namespace infini

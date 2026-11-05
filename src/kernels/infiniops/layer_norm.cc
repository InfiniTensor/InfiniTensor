#include "infiniops_common.h"
#include "infiniops_dispatch.h"
#include "operators/layer_norm.h"

#include <base/layer_norm.h>

namespace infini {
namespace {

class LayerNormInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op, const RuntimeObj *context) const override {
        auto op = as<LayerNormObj>(_op);
        IT_ASSERT(op->getStashType() == 1, "Only FP32 LayerNormalization stash is supported");
        auto input = infiniops::toInfiniOpsTensor(op->getInputs(0), context);
        auto scale = infiniops::toInfiniOpsTensor(op->getInputs(1), context);
        auto output = infiniops::toInfiniOpsTensor(op->getOutput(), context);
        std::optional<::infini::ops::Tensor> bias;
        if (auto value = op->getBias())
            bias.emplace(infiniops::toInfiniOpsTensor(value, context));
        auto handle = infiniops::makeInfiniOpsHandle(context);
        auto config = infiniops::makeInfiniOpsConfig<::infini::ops::LayerNorm>(context);
        const auto axis = static_cast<size_t>(op->getAxis());
        IT_ASSERT(axis < input.ndim());
        if (axis + 1 != input.ndim()) {
            // The InfiniOps interface normalizes the last dimension.
            using Shape = ::infini::rt::TensorView::Shape;
            Shape shape;
            shape.resize(axis + 1);
            for (size_t i = 0; i < axis; ++i)
                shape[i] = input.shape()[i];
            shape[axis] = scale.numel();
            auto flatten = [&](::infini::ops::Tensor value,
                               const Shape &target) {
                IT_ASSERT(value.IsContiguous());
                return infiniops::makeInfiniOpsTensor(value.data(), target,
                                                      value.dtype(), context);
            };
            const Shape weightShape{shape[axis]};
            auto flatBias = bias ? std::optional{flatten(*bias, weightShape)}
                                 : std::nullopt;
            infiniops::dispatch::callLayerNorm(
                handle, config, flatten(input, shape), flatten(scale, weightShape),
                flatBias, op->getEps(), flatten(output, shape));
            return;
        }
        infiniops::dispatch::callLayerNorm(handle, config, input, scale, bias,
                                          op->getEps(), output);
    }
};

} // namespace

REGISTER_KERNEL(ExecutionProvider::Infini, OpType::LayerNormalization,
                LayerNormInfiniOps, "LayerNorm_InfiniOps");
} // namespace infini

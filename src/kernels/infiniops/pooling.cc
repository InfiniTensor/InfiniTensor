#include "operators/pooling.h"
#include "infiniops_common.h"
#include "infiniops_dispatch.h"

#ifdef USE_INFINIOPS_ATEN_KERNELS
#include <base/avg_pool2d.h>
#include <base/max_pool2d_with_indices.h>
#endif
#include <cstdint>
#include <optional>
#include <vector>

namespace infini {
namespace {

#ifdef USE_INFINIOPS_ATEN_KERNELS
std::vector<int64_t> poolingValues(int first, int second) {
    return {static_cast<int64_t>(first), static_cast<int64_t>(second)};
}

class PoolingAtenInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<PoolingObj>(_op);
        auto input = infiniops::toInfiniOpsTensor(op->getInputs(0), context);
        auto output = infiniops::toInfiniOpsTensor(op->getOutput(), context);
        const auto [ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
        auto handle = infiniops::makeInfiniOpsHandle(context);
        if (op->getOpType() == OpType::AveragePool) {
            auto config =
                infiniops::makeInfiniOpsAtenConfig<::infini::ops::AvgPool2d>(
                    context);
            IT_ASSERT(dh == 1 && dw == 1,
                      "ATen AvgPool does not support dilation");
            infiniops::dispatch::callAvgPool2d(
                handle, config, input, poolingValues(op->getKh(), op->getKw()),
                poolingValues(sh, sw), poolingValues(ph, pw),
                op->getCeilMode() != 0, true, std::optional<int64_t>{}, output);
            return;
        }

        IT_ASSERT(op->getOpType() == OpType::MaxPool);
        const auto indexBytes = op->getOutput()->size() * sizeof(int64_t);
        auto indexBlob = infiniops::allocTemporaryBlob(context, indexBytes);
        ::infini::rt::TensorView::Shape indexShape;
        for (const auto dim : op->getOutput()->getDims()) {
            indexShape.push_back(
                static_cast<::infini::rt::TensorView::Size>(dim));
        }
        auto indices = infiniops::makeInfiniOpsTensor(
            indexBlob->getPtr<void *>(), indexShape,
            ::infini::rt::DataType::kInt64, context);
        auto config = infiniops::makeInfiniOpsAtenConfig<
            ::infini::ops::MaxPool2dWithIndices>(context);
        infiniops::dispatch::callMaxPool2dWithIndices(
            handle, config, input, poolingValues(op->getKh(), op->getKw()),
            poolingValues(sh, sw), poolingValues(ph, pw), poolingValues(dh, dw),
            op->getCeilMode() != 0, output, indices);
    }
};
#endif

} // namespace

#ifdef USE_INFINIOPS_ATEN_KERNELS
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::AveragePool,
                PoolingAtenInfiniOps, "AveragePool_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::MaxPool,
                PoolingAtenInfiniOps, "MaxPool_InfiniOps");
#endif

} // namespace infini

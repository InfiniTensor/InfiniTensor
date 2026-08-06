#include "operators/batch_norm.h"
#include "infiniops_common.h"
#include "infiniops_dispatch.h"

#ifdef USE_INFINIOPS_ATEN_KERNELS
#include <base/native_batch_norm.h>
#endif
#include <cstddef>
#include <optional>

namespace infini {
namespace {

#ifdef USE_INFINIOPS_ATEN_KERNELS
class BatchNormAtenInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<BatchNormObj>(_op);
        IT_ASSERT(!op->getTrainingMode());

        auto input = infiniops::toInfiniOpsTensor(op->getInputs(0), context);
        auto mean = infiniops::toInfiniOpsTensor(op->getInputs(1), context);
        auto var = infiniops::toInfiniOpsTensor(op->getInputs(2), context);
        auto scale = infiniops::toInfiniOpsTensor(op->getInputs(3), context);
        auto bias = infiniops::toInfiniOpsTensor(op->getInputs(4), context);
        auto output = infiniops::toInfiniOpsTensor(op->getOutput(), context);

        const auto channels = op->getInputs(1)->size();
        const auto savedBytes = channels * sizeof(float);
        auto savedMeanBlob = infiniops::allocTemporaryBlob(context, savedBytes);
        auto savedInvStdBlob =
            infiniops::allocTemporaryBlob(context, savedBytes);
        ::infini::rt::TensorView::Shape savedShape{channels};
        auto savedMean = infiniops::makeInfiniOpsTensor(
            savedMeanBlob->getPtr<void *>(), savedShape,
            ::infini::rt::DataType::kFloat32, context);
        auto savedInvStd = infiniops::makeInfiniOpsTensor(
            savedInvStdBlob->getPtr<void *>(), savedShape,
            ::infini::rt::DataType::kFloat32, context);

        auto handle = infiniops::makeInfiniOpsHandle(context);
        auto config =
            infiniops::makeInfiniOpsAtenConfig<::infini::ops::NativeBatchNorm>(
                context);
        infiniops::dispatch::callNativeBatchNorm(
            handle, config, input, std::optional<::infini::ops::Tensor>{scale},
            std::optional<::infini::ops::Tensor>{bias},
            std::optional<::infini::ops::Tensor>{mean},
            std::optional<::infini::ops::Tensor>{var}, false,
            static_cast<double>(op->getMomentum()),
            static_cast<double>(op->getEps()), output, savedMean, savedInvStd);
    }
};
#endif

} // namespace

#ifdef USE_INFINIOPS_ATEN_KERNELS
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::BatchNormalization,
                BatchNormAtenInfiniOps, "BatchNorm_InfiniOps");
#endif

} // namespace infini

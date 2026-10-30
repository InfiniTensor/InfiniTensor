#include "execution.h"
#include "operators/rms_norm.h"
#include "tensor_view.h"
namespace infini {
class NativeRmsNormCompatibility final : public CpuKernelWithoutConfig {
    void compute(const Operator &op, const RuntimeObj *runtime) const override {
        const auto noopWorkspace =
            [](size_t) -> std::shared_ptr<infiniops::Workspace> {
            IT_ASSERT(false, "CPU RMSNorm unexpectedly requires workspace");
            return {};
        };
        const auto prepared = infiniops::Registry::instance().prepare(
            "RmsNorm",
            infiniops::Values{
                infiniops::toInfiniOpsTensor(op->getInputs(0), runtime),
                infiniops::toInfiniOpsTensor(op->getInputs(1), runtime),
                double(1e-6),
                infiniops::toInfiniOpsTensor(op->getOutput(), runtime)},
            noopWorkspace);
        prepared.run(::infini::ops::Handle{});
    }
};
REGISTER_KERNEL(ExecutionProvider::NativeCpu, OpType::RMSNorm,
                NativeRmsNormCompatibility, "RMSNorm_InfiniOps_CPU");
} // namespace infini

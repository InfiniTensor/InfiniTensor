#pragma once

#include "core/kernel.h"
#include "core/operator.h"

namespace infini {

/// A wrapper kernel that copies GPU tensors to CPU, runs the original CPU
/// kernel, then copies the result back to GPU. Used for operators that
/// InfiniOps does not (yet) support on GPU devices.
///
/// For now this is a placeholder — the actual device-specific memory
/// copy will be added when GPU backends are enabled.
class GpuToCpuFallback : public KernelWithoutConfig {
  public:
    GpuToCpuFallback(Kernel *cpuKernel) : cpuKernel_(cpuKernel) {}
    ~GpuToCpuFallback() override { delete cpuKernel_; }

    void compute(const Operator &op,
                 const RuntimeObj *context) const override {
        // For CPU devices, delegate directly to the CPU kernel.
        // For GPU devices, this would need GPU→CPU→GPU memory copies.
        cpuKernel_->compute(op, context);
    }

  private:
    Kernel *cpuKernel_;
};

} // namespace infini

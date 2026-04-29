#pragma once

#include "core/kernel.h"

namespace infini {

// Base class for InfiniOps adapter kernels.
// Subclasses implement compute() by converting InfiniTensor tensors to
// InfiniOps tensors and calling the corresponding InfiniOps operator.
class InfiniOpsAdapterKernel : public CpuKernelWithoutConfig {
  public:
    virtual ~InfiniOpsAdapterKernel() = default;
};

} // namespace infini

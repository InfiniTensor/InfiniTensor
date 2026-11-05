#pragma once

#include "cuda/cuda_runtime.h"

namespace infini {

class CorexRuntimeObj : public CudaRuntimeObj {
  public:
    explicit CorexRuntimeObj(int deviceId = 0)
        : CudaRuntimeObj(deviceId, Device::COREX) {}

    string toString() const override;
};

} // namespace infini
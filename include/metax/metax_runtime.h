#pragma once

#include "cuda/cuda_runtime.h"

namespace infini {

class MetaxRuntimeObj : public CudaRuntimeObj {
  public:
    explicit MetaxRuntimeObj(int deviceId = 0)
        : CudaRuntimeObj(deviceId, Device::METAX) {}

    string toString() const override;
};

} // namespace infini

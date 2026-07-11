#pragma once

#include "vendor/cuda_compatible_runtime.h"

namespace infini {

class MetaxRuntimeObj final : public CudaCompatibleRuntimeObj {
  public:
    explicit MetaxRuntimeObj(int deviceId = 0)
        : CudaCompatibleRuntimeObj(Device::METAX, deviceId, "MetaX") {}
};

} // namespace infini

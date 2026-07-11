#pragma once

#include "vendor/cuda_compatible_runtime.h"

namespace infini {

class IluvatarRuntimeObj final : public CudaCompatibleRuntimeObj {
  public:
    explicit IluvatarRuntimeObj(int deviceId = 0)
        : CudaCompatibleRuntimeObj(Device::ILUVATAR, deviceId, "Iluvatar") {}
};

} // namespace infini

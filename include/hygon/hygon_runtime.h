#pragma once

#include "vendor/cuda_compatible_runtime.h"

namespace infini {

class HygonRuntimeObj final : public CudaCompatibleRuntimeObj {
  public:
    explicit HygonRuntimeObj(int deviceId = 0)
        : CudaCompatibleRuntimeObj(Device::HYGON, deviceId, "Hygon") {}

    void initComm(const string &, int, int) override {
        IT_TODO_HALT_MSG("Hygon collectives require a future InfiniCCL backend");
    }
};

} // namespace infini

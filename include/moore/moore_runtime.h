#pragma once

#include "core/runtime.h"
#ifdef USE_INFINICCL
#include "communication/infiniccl_communicator.h"
#endif

#include <musa_runtime.h>
#include <memory>

namespace infini {

inline void checkMooreRuntime(musaError_t result, const string &operation) {
    IT_ASSERT(result == musaSuccess, operation + ": " + musaGetErrorString(result));
}

class MooreRuntimeObj final : public SdkRuntimeObj {
  public:
    explicit MooreRuntimeObj(int deviceId = 0)
        : SdkRuntimeObj(Device::MOORE, deviceId) {
        checkMooreRuntime(musaSetDevice(deviceId), "musaSetDevice");
    }

    void *alloc(size_t size) override {
        void *ptr = nullptr;
        checkMooreRuntime(musaMalloc(&ptr, size), "musaMalloc");
        return ptr;
    }

    void dealloc(void *ptr) override {
        if (ptr != nullptr) checkMooreRuntime(musaFree(ptr), "musaFree");
    }

    void copyBlobFromCPU(void *dst, const void *src,
                         size_t bytes) const override {
        checkMooreRuntime(musaMemcpy(dst, src, bytes, musaMemcpyHostToDevice),
                          "musaMemcpy host-to-device");
    }

    void copyBlobToCPU(void *dst, const void *src,
                       size_t bytes) const override {
        checkMooreRuntime(musaMemcpy(dst, src, bytes, musaMemcpyDeviceToHost),
                          "musaMemcpy device-to-host");
    }

    void copyBlobInsideRuntime(void *dst, const void *src,
                               size_t bytes) const override {
        checkMooreRuntime(musaMemcpy(dst, src, bytes, musaMemcpyDeviceToDevice),
                          "musaMemcpy device-to-device");
    }

    void sync() const override {
        checkMooreRuntime(musaDeviceSynchronize(), "musaDeviceSynchronize");
    }

    string toString() const override { return "Moore Runtime"; }

    void initComm(const string &name, int worldSize, int rank) override {
        IT_ASSERT(worldSize > 0 && rank >= 0 && rank < worldSize);
        IT_ASSERT(!comm, "communicator is already initialized");
#ifdef USE_INFINICCL
        comm = std::make_unique<InfiniCclCommunicatorObj>(name, worldSize, rank);
#else
        IT_TODO_HALT_MSG("Not compiled with InfiniCCL");
#endif
    }

    CommunicatorObj &getCommunicator() const override {
        IT_ASSERT(comm != nullptr, "communicator is not initialized");
        return *comm;
    }

  private:
    std::unique_ptr<CommunicatorObj> comm;
};

} // namespace infini

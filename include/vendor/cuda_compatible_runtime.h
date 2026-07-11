#pragma once

#include "core/runtime.h"
#ifdef USE_INFINICCL
#include "communication/infiniccl_communicator.h"
#endif

#include <cuda_runtime.h>
#include <memory>
#include <utility>

namespace infini {

inline void checkCudaCompatibleRuntime(cudaError_t result,
                                       const string &operation) {
    IT_ASSERT(result == cudaSuccess,
              operation + ": " + cudaGetErrorString(result));
}

class CudaCompatibleRuntimeObj : public SdkRuntimeObj {
  public:
    CudaCompatibleRuntimeObj(Device device, int deviceId, string runtimeName)
        : SdkRuntimeObj(device, deviceId), runtimeName(std::move(runtimeName)) {
        checkCudaCompatibleRuntime(cudaSetDevice(deviceId), "cudaSetDevice");
    }

    void *alloc(size_t size) override {
        void *ptr = nullptr;
        checkCudaCompatibleRuntime(cudaMalloc(&ptr, size), "cudaMalloc");
        return ptr;
    }

    void dealloc(void *ptr) override {
        if (ptr != nullptr) {
            checkCudaCompatibleRuntime(cudaFree(ptr), "cudaFree");
        }
    }

    void copyBlobFromCPU(void *dst, const void *src,
                         size_t bytes) const override {
        checkCudaCompatibleRuntime(
            cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy host-to-device");
    }

    void copyBlobToCPU(void *dst, const void *src,
                       size_t bytes) const override {
        checkCudaCompatibleRuntime(
            cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy device-to-host");
    }

    void copyBlobInsideRuntime(void *dst, const void *src,
                               size_t bytes) const override {
        checkCudaCompatibleRuntime(
            cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice),
            "cudaMemcpy device-to-device");
    }

    void sync() const override {
        checkCudaCompatibleRuntime(cudaDeviceSynchronize(),
                                   "cudaDeviceSynchronize");
    }

    string toString() const override { return runtimeName + " Runtime"; }

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
    string runtimeName;
    std::unique_ptr<CommunicatorObj> comm;
};

} // namespace infini

#include "core/infiniops_bridge/infiniops_runtime.h"
#include <cstdlib>
#include <cstring>

#ifdef WITH_NVIDIA
#include <cuda_runtime.h>
#endif

namespace infini {

InfiniOpsRuntimeObj::InfiniOpsRuntimeObj(Device device)
    : RuntimeObj(device) {
    switch (device.type()) {
    case Device::Type::kCpu:
        workspace_ = std::malloc(workspaceSize_);
        break;
#ifdef WITH_NVIDIA
    case Device::Type::kNvidia:
        cudaMalloc(&workspace_, workspaceSize_);
        break;
#endif
    default:
        // For unsupported devices, allocate on CPU as fallback.
        workspace_ = std::malloc(workspaceSize_);
        break;
    }
}

void *InfiniOpsRuntimeObj::alloc(size_t size) {
    void *ptr = nullptr;
    switch (device.type()) {
    case Device::Type::kCpu:
        ptr = std::malloc(size);
        break;
#ifdef WITH_NVIDIA
    case Device::Type::kNvidia:
        cudaMalloc(&ptr, size);
        break;
#endif
    default:
        ptr = std::malloc(size);
        break;
    }
    return ptr;
}

void InfiniOpsRuntimeObj::dealloc(void *ptr) {
    if (ptr == nullptr)
        return;
    switch (device.type()) {
    case Device::Type::kCpu:
        std::free(ptr);
        break;
#ifdef WITH_NVIDIA
    case Device::Type::kNvidia:
        cudaFree(ptr);
        break;
#endif
    default:
        std::free(ptr);
        break;
    }
}

string InfiniOpsRuntimeObj::toString() const {
    return "InfiniOps Runtime (" + device.ToString() + ")";
}

void *InfiniOpsRuntimeObj::getWorkspace(size_t size) {
    if (size > workspaceSize_) {
        IT_TODO_HALT_MSG("Workspace size exceeded");
    }
    void *ptr = static_cast<uint8_t *>(workspace_) + workspaceCursor_;
    workspaceCursor_ += size;
    // Align to 256 bytes
    workspaceCursor_ = (workspaceCursor_ + 255) & ~size_t(255);
    return ptr;
}

void InfiniOpsRuntimeObj::resetWorkspace() { workspaceCursor_ = 0; }

} // namespace infini

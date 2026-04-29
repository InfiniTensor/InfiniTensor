#pragma once

#include "core/runtime.h"

namespace infini {

// Unified runtime object that uses InfiniOps for operator execution.
// Inherits graph execution (run()) and blob copy (memcpy) from CpuRuntimeObj.
// Overrides alloc/dealloc to support device-specific memory allocation.
class InfiniOpsRuntimeObj : public CpuRuntimeObj {
  public:
    explicit InfiniOpsRuntimeObj(Device device);

    void *alloc(size_t size) override;
    void dealloc(void *ptr) override;

    string toString() const override;

    // Workspace management for InfiniOps operators that need scratch memory.
    void *getWorkspace(size_t size);
    void resetWorkspace();

  private:
    // Pre-allocated workspace buffer (7GB, matching original CUDA runtime).
    static constexpr size_t kDefaultWorkspaceSize = 7ull << 30;
    size_t workspaceSize_ = kDefaultWorkspaceSize;
    size_t workspaceCursor_ = 0;
    void *workspace_ = nullptr;
};

} // namespace infini

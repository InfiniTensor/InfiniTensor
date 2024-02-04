#pragma once
// #include "core/ref.h"
#include "core/runtime.h"

namespace infini {

// template <typename T> using Workspace = Ref<WorkspaceObj<T>>;

template <class T> class WorkspaceObj {
  private:
    T workspace;           // workspace pointer
    size_t workspaceSize;  // Size of workspace
    size_t workspaceAlloc; // currently use workspace size

  public:
    WorkspaceObj(T workspace_, size_t workspaceSize_)
        : workspace(workspace_), workspaceSize(workspaceSize_) {
        workspaceAlloc = 0;
    }
    virtual ~WorkspaceObj() {
        // Dealloc workspace in RuntimeObj
        // Set workspace = nullptr here
        workspace = nullptr;
    }
    size_t getWorkspaceSize() const { return workspaceSize; }

    T getWorkspace(size_t size) {
        // Get unused workspace
        IT_ASSERT(size + workspaceAlloc <= workspaceSize);
        auto ret = (T)(static_cast<uint8_t *>(workspace) + workspaceAlloc);
        workspaceAlloc += size;
        return ret;
    }
    T getWorkspace() {
        // Override getWorkspace in order to dealloc in runtime
        return workspace;
    }
    void resetWorkspace() {
        // Reset workspaceAlloc every time end kernel
        workspaceAlloc = 0;
    }
    size_t getWorkspaceAlloc() const { return workspaceAlloc; }
};

} // namespace infini

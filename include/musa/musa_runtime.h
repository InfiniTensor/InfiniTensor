#pragma once
#include "core/runtime.h"
#include "musa/musa_common.h"

namespace infini {

class MusaRuntimeObj : public RuntimeObj {
  private:
    MusaPtr workspace;
    size_t workspaceSize;

  public:
    explicit MusaRuntimeObj(int deviceId = 0)
        : RuntimeObj(Device::MUSA, deviceId) {
        checkMusaError(musaSetDevice(deviceId));
        workspaceSize = 7ll << 30; // 7GB
        workspace = alloc(workspaceSize);
    }
    virtual ~MusaRuntimeObj() {
        try {
            dealloc(workspace);
        } catch (const std::exception &e) {
            std::cerr << "Error in ~MusaRuntimeObj: " << e.what() << std::endl;
        }
    }
    string toString() const override;
    void run(const Graph &graph, bool tune = false,
             bool profiling = false) const;
    void sync() const;
    MusaPtr alloc(size_t size) override {
        void *ptr;
        checkMusaError(musaMalloc(&ptr, size));
        // printf("musa malloc: %p %lu bytes\n", ptr, size);
        return ptr;
    }
    void dealloc(void *ptr) override { checkMusaError(musaFree(ptr)); }
    size_t getWorkspaceSize() const { return workspaceSize; }
    MusaPtr getWorkspace(size_t size) const {
        IT_ASSERT(size <= workspaceSize);
        return workspace;
    }

    void copyBlobFromCPU(void *dst, const void *src,
                         size_t bytes) const override {
        checkMusaError(musaMemcpy(dst, src, bytes, musaMemcpyHostToDevice));
    }

    void copyBlobToCPU(void *dst, const void *src,
                       size_t bytes) const override {
        checkMusaError(musaMemcpy(dst, src, bytes, musaMemcpyDeviceToHost));
    }

    void copyBlobInsideRuntime(void *dst, const void *src,
                               size_t bytes) const override {
        checkMusaError(musaMemcpy(dst, src, bytes, musaMemcpyDeviceToDevice));
    }

    void runWithoutSync(const Graph &graph) const;
};

} // namespace infini

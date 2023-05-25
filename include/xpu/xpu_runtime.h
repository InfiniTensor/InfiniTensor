#pragma once
#include "xpu/xpu_common.h"
#include "core/runtime.h"

namespace infini {

class XPURuntimeObj : public RuntimeObj {
  private:
    baidu::xpu::api::Context* xdnn;
    XPUPtr workspace;
    size_t workspaceSize;

  public:
    XPURuntimeObj() : RuntimeObj(Device::XPU) {
	xdnn = baidu::xpu::api::create_context(); 
        // 10GB for Longformer
        // size_t longformerNum = 3lu * (1 << 30);
        workspaceSize = 7ll << 30; // 7 GB
        workspace = alloc(workspaceSize);
    }
    virtual ~XPURuntimeObj() {
        dealloc(workspace);
	baidu::xpu::api::destroy_context(xdnn);
    }
    string toString() const override;

    void run(const Graph &graph, bool tune = false,
             bool profiling = false) const;
    // double runEvaluation(const Graph &graph, int nWarmups,
    //                      int nEvaluations) const;
    void sync() const;
    XPUPtr alloc(size_t size) override {
        void *ptr;
	xpu_malloc(&ptr, size);
        return ptr;
    }
    void dealloc(void *ptr) override { xpu_free(ptr); }
    baidu::xpu::api::Context* XPUHandle() const { return xdnn; }
    XPUPtr getWorkspace(size_t size) const {
        IT_ASSERT(size <= workspaceSize);
        return workspace;
    }

    void copyBlobFromCPU(void *dst, const void *src,
                         size_t bytes) const override {
	    xpu_memcpy(dst, const_cast<void *>(src), bytes, XPUMemcpyKind::XPU_HOST_TO_DEVICE);
    }

    void copyBlobToCPU(void *dst, const void *src,
                       size_t bytes) const override {
	    xpu_memcpy(dst, const_cast<void *>(src), bytes, XPUMemcpyKind::XPU_DEVICE_TO_HOST);
    }

    void copyBlobInsideRuntime(void *dst, const void *src,
                               size_t bytes) const override {
            xpu_memcpy(dst, const_cast<void *>(src), bytes, XPUMemcpyKind::XPU_DEVICE_TO_DEVICE);
    }

  private:
    void runWithoutSync(const Graph &graph, bool tune, bool profiling) const;
};

} // namespace infini

#pragma once
#include "core/runtime.h"
#include "xpu/xpu_common.h"

namespace infini {

class XPURuntimeObj : public RuntimeObj {
  private:
    baidu::xpu::api::Context *xdnn;
    XPUPtr workspace;
    size_t workspaceSize;

  public:
    XPURuntimeObj() : RuntimeObj(Device::XPU) {
        xdnn = baidu::xpu::api::create_context();
        // 10GB for Longformer
        // size_t longformerNum = 3lu * (1 << 30);
        workspaceSize = 3ll << 30; // 3 GB
        // std::cout<<workspaceSize/1024/1024/1024<< std::endl;
        // std::cout<<std::bitset<64>(workspaceSize)<< std::endl;
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
        checkXPUError(
            xpu_malloc_ex((void **)&ptr, size, XPUMemoryKind::XPU_MEM_MAIN));
        return ptr;
    }
    void dealloc(void *ptr) override { xpu_free(ptr); }
    baidu::xpu::api::Context *XPUHandle() const { return xdnn; }
    XPUPtr getWorkspace(size_t size) const {
        IT_ASSERT(size <= workspaceSize);
        return workspace;
    }

    void copyBlobFromCPU(void *dst, const void *src,
                         size_t bytes) const override {
        xpu_memcpy(dst, const_cast<void *>(src), bytes,
                   XPUMemcpyKind::XPU_HOST_TO_DEVICE);
    }

    void copyBlobToCPU(void *dst, const void *src,
                       size_t bytes) const override {
        xpu_memcpy(dst, const_cast<void *>(src), bytes,
                   XPUMemcpyKind::XPU_DEVICE_TO_HOST);
    }

    void copyBlobInsideRuntime(void *dst, const void *src,
                               size_t bytes) const override {
        xpu_memcpy(dst, const_cast<void *>(src), bytes,
                   XPUMemcpyKind::XPU_DEVICE_TO_DEVICE);
    }

    void initComm(const string &, int, int) override { IT_TODO_HALT(); }

    CommunicatorObj &getCommunicator() const override { IT_TODO_HALT(); }

  private:
    void runWithoutSync(const Graph &graph, bool tune, bool profiling) const;
};

} // namespace infini

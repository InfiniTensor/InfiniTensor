#pragma once
#include "core/runtime.h"
#include "kunlun/kunlun_common.h"

namespace infini {

class KUNLUNRuntimeObj : public RuntimeObj {
  private:
    baidu::xpu::api::Context *xdnn;
    KUNLUNPtr workspace;
    size_t workspaceSize;

  public:
    KUNLUNRuntimeObj() : RuntimeObj(Device::KUNLUN) {
        xdnn = baidu::xpu::api::create_context();
        // 10GB for Longformer
        // size_t longformerNum = 3lu * (1 << 30);
        workspaceSize = 3ll << 30; // 3 GB
        // std::cout<<workspaceSize/1024/1024/1024<< std::endl;
        // std::cout<<std::bitset<64>(workspaceSize)<< std::endl;
        workspace = alloc(workspaceSize);
    }
    virtual ~KUNLUNRuntimeObj() {
        dealloc(workspace);
        baidu::xpu::api::destroy_context(xdnn);
    }
    string toString() const override;

    void run(const Graph &graph, bool tune = false,
             bool profiling = false) const;
    // double runEvaluation(const Graph &graph, int nWarmups,
    //                      int nEvaluations) const;
    void sync() const;
    KUNLUNPtr alloc(size_t size) override {
        void *ptr;
        checkKUNLUNError(
            xpu_malloc((void **)&ptr, size));
        return ptr;
    }
    void dealloc(void *ptr) override { xpu_free(ptr); }
    baidu::xpu::api::Context *KUNLUNHandle() const { return xdnn; }
    KUNLUNPtr getWorkspace(size_t size) const {
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

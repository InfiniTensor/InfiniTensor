#pragma once
#include "core/runtime.h"
#include "core/workspace.h"
#include "kunlun/kunlun_common.h"
#ifdef INFINI_USE_XCCL
#include "kunlun/xccl_communicator.h"
#endif
namespace infini {

class KUNLUNRuntimeObj : public RuntimeObj {
  private:
    xdnn::Context *ctx;
    std::unique_ptr<CommunicatorObj> comm;
    // KUNLUNPtr workspace;
    // size_t workspaceSize;
    Workspace<KUNLUNPtr> workspace;

  public:
    KUNLUNRuntimeObj(int deviceId = 0) : RuntimeObj(Device::KUNLUN) {
        xpu_set_device(deviceId);
        ctx = xdnn::create_context();
        // 10GB for Longformer
        // size_t longformerNum = 3lu * (1 << 30);
        size_t workspaceSize = 3llu << 30; // 3 GB
        KUNLUNPtr wkspacePtr = alloc(workspaceSize);
        workspace =
            make_ref<WorkspaceObj<KUNLUNPtr>>(wkspacePtr, workspaceSize);
    }
    virtual ~KUNLUNRuntimeObj() {
        KUNLUNPtr wkspacePtr = workspace->getWorkspace();
        dealloc(wkspacePtr);
        xdnn::destroy_context(ctx);
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
            xpu_malloc_ex((void **)&ptr, size, XPUMemoryKind::XPU_MEM_MAIN));
        return ptr;
    }
    void dealloc(void *ptr) override { xpu_free(ptr); }

    xdnn::Context *KUNLUNHandle() const { return ctx; }
    // Get $size workspace by bytes
    KUNLUNPtr getWorkspace(size_t size) const {
        auto ret = workspace->getWorkspace(size);
        return ret;
    }
    Workspace<KUNLUNPtr> getWorkspaceObj() const { return workspace; }

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
    void initComm(const string &name, int worldSize, int rank) final;

    CommunicatorObj &getCommunicator() const final { return *comm; }

  private:
    void runWithoutSync(const Graph &graph, bool tune, bool profiling) const;
};

} // namespace infini

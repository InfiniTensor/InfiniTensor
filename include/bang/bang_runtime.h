#pragma once
#include "core/runtime.h"
#include "bang/bang_common.h"

namespace infini {

class BangRuntimeObj : public RuntimeObj {
  private:
    cnnlHandle_t cnnl;
    BangPtr workspace;
    size_t workspaceSize;

  public:
    BangRuntimeObj() : RuntimeObj(Device::BANG) {
        checkCnnlError(cnnlCreate(&cnnl));
        // 10GB for Longformer
        // size_t longformerNum = 3lu * (1 << 30);
        workspaceSize = 7ll << 30; // 7 GB
        workspace = alloc(workspaceSize);
    }
    virtual ~BangRuntimeObj() {
        dealloc(workspace);
        checkCnnlError(cnnlDestroy(cnnl));
    }

    void run(const Graph &graph, bool tune = false,
             bool profiling = false) const;
    // double runEvaluation(const Graph &graph, int nWarmups,
    //                      int nEvaluations) const;
    void sync() const;
    BangPtr alloc(size_t size) override {
        void *ptr;
        checkBangError(cnrtMalloc(&ptr, size));
        return ptr;
    }
    void dealloc(void *ptr) override { checkBangError(cnrtFree(ptr)); }
    cnnlHandle_t cnnlHandle() const { return cnnl; }
    BangPtr getWorkspace(size_t size) const {
        IT_ASSERT(size <= workspaceSize);
        return workspace;
    }

    void copyBlobFromCPU(void *dst, void *src, size_t bytes) const override {
        checkBangError(cnrtMemcpy(dst, src, bytes, CNRT_MEM_TRANS_DIR_HOST2DEV));
    }

    void copyBlobToCPU(void *dst, void *src, size_t bytes) const override {
        checkBangError(cnrtMemcpy(dst, src, bytes, CNRT_MEM_TRANS_DIR_DEV2HOST));
    }

    void copyBlobInsideRuntime(void *dst, void *src,
                               size_t bytes) const override {
        checkBangError(cnrtMemcpy(dst, src, bytes, CNRT_MEM_TRANS_DIR_PEER2PEER));
    }

  private:
    void runWithoutSync(const Graph &graph, bool tune, bool profiling) const;
};
} // namespace infini

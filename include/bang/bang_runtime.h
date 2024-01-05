#pragma once
#include "bang/bang_common.h"
#include "core/runtime.h"

namespace infini {

class BangRuntimeObj : public RuntimeObj {
  private:
    cnnlHandle_t cnnl;
    cnrtQueue_t queue;
    std::unique_ptr<CommunicatorObj> comm;
    BangPtr workspace;
    size_t workspaceSize;
    mutable size_t cursor;

  public:
    explicit BangRuntimeObj(int deviceId = 0)
        : RuntimeObj(Device::BANG, deviceId) {
        cnInit(0);
        CNdev dev;
        cnDeviceGet(&dev, deviceId);
        checkBangError(cnrtSetDevice(dev));
        checkBangError(cnrtQueueCreate(&queue));

        checkCnnlError(cnnlCreate(&cnnl));
        checkCnnlError(cnnlSetQueue(cnnl, queue));
        // 10GB for Longformer
        // size_t longformerNum = 3lu * (1 << 30);
        workspaceSize = 7ll << 30; // 7 GB
        cursor = 0;
        workspace = alloc(workspaceSize);
    }
    virtual ~BangRuntimeObj() {
        dealloc(workspace);
        checkBangError(cnrtQueueDestroy(queue));
        checkCnnlError(cnnlDestroy(cnnl));
    }
    string toString() const override;

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
        IT_ASSERT((cursor + size) <= workspaceSize);
        cursor += size;
        void *temp = workspace;
        temp += (cursor - size);
        return temp;
    }

    void resetWorkspace() const { cursor = 0; }

    void copyBlobFromCPU(void *dst, const void *src,
                         size_t bytes) const override {
        checkBangError(cnrtMemcpy(dst, const_cast<void *>(src), bytes,
                                  CNRT_MEM_TRANS_DIR_HOST2DEV));
    }

    void copyBlobToCPU(void *dst, const void *src,
                       size_t bytes) const override {
        checkBangError(cnrtMemcpy(dst, const_cast<void *>(src), bytes,
                                  CNRT_MEM_TRANS_DIR_DEV2HOST));
    }

    void copyBlobInsideRuntime(void *dst, const void *src,
                               size_t bytes) const override {
        checkBangError(cnrtMemcpy(dst, const_cast<void *>(src), bytes,
                                  CNRT_MEM_TRANS_DIR_PEER2PEER));
    }
    void initComm(const string &name, int worldSize, int rank) final;
    CommunicatorObj &getCommunicator() const override { return *comm; }
    cnrtQueue_t getBangQueue() const { return queue; }

  private:
    void runWithoutSync(const Graph &graph, bool tune, bool profiling) const;
};

} // namespace infini

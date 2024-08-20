#pragma once
#include "ascend/ascend_common.h"
#include "core/runtime.h"

namespace infini {

class ASCENDRuntimeObj : public RuntimeObj {
  private:
    aclrtContext context;
    aclrtStream stream;
    std::unique_ptr<CommunicatorObj> comm;
    ASCENDPtr workspace = nullptr;
    uint64_t workspaceSize;

  public:
    ASCENDRuntimeObj(int deviceId = 0) : RuntimeObj(Device::ASCEND, deviceId) {
        // auto ret = aclInit(nullptr);
        // CHECK_RET(ret == ACL_SUCCESS,
        //           LOG_PRINT("aclInit failed. ERROR: %d\n", ret));
        auto ret = aclrtSetDevice(deviceId);
        checkASCENDError(ret);
        ret = aclrtCreateStream(&stream);
        checkASCENDError(ret);

        // 10GB for Longformer
        // size_t longformerNum = 3lu * (1 << 30);
        workspaceSize = 3ll * (1 << 30); // 3 GB

        workspace = alloc(workspaceSize);
    }
    virtual ~ASCENDRuntimeObj() {
        dealloc(workspace);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        // aclFinalize();
    }
    string toString() const override;

    void run(const Graph &graph, bool tune = false,
             bool profiling = false) const;

    void sync() const;
    ASCENDPtr alloc(size_t size) override {
        void *ptr;
        checkASCENDError(
            aclrtMalloc((void **)&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
        return ptr;
    }
    void dealloc(void *ptr) override { checkASCENDError(aclrtFree(ptr)); }
    aclrtStream ASCENDHandle() const { return stream; }
    ASCENDPtr getWorkspace(uint64_t size) const {
        IT_ASSERT(size <= workspaceSize);
        return workspace;
    }

    void copyBlobFromCPU(void *dst, const void *src,
                         size_t bytes) const override {
        aclrtMemcpy(dst, bytes, const_cast<void *>(src), bytes,
                    ACL_MEMCPY_HOST_TO_DEVICE);
    }

    void copyBlobToCPU(void *dst, const void *src,
                       size_t bytes) const override {
        aclrtMemcpy(dst, bytes, const_cast<void *>(src), bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST);
    }

    void copyBlobInsideRuntime(void *dst, const void *src,
                               size_t bytes) const override {
        aclrtMemcpy(dst, bytes, const_cast<void *>(src), bytes,
                    ACL_MEMCPY_DEVICE_TO_DEVICE);
    }

    void initComm(const string &name, int worldSize, int rank) final;

    CommunicatorObj &getCommunicator() const override { return *comm; }

  private:
    void runWithoutSync(const Graph &graph, bool tune, bool profiling) const;
};

} // namespace infini

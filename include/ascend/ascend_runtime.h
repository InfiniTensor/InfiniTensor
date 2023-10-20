#pragma once
#include "core/runtime.h"
#include "ascend/ascend_common.h"


#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

namespace infini {

class ASCENDRuntimeObj : public RuntimeObj {
  private:
    aclrtContext aclnn;
    aclrtStream stream;
    ASCENDPtr workspace;
    size_t workspaceSize;

  public:
    ASCENDRuntimeObj(int deviceId = 0)
	    : RuntimeObj(Device::ASCEND, deviceId) {
        std::cout<<"1------------------------------------"<< std::endl;
	auto ret = aclrtSetDevice(deviceId);
        std::cout<<"2------------------------------------"<< std::endl;
	CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret));
        ret = aclrtCreateContext(&aclnn, deviceId);
	CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret));
	ret = aclrtSetCurrentContext(aclnn);
	CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret));
        ret = aclrtCreateStream(&stream);
	CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret));
	ret = aclInit(nullptr);
	CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret));
        // 10GB for Longformer
        // size_t longformerNum = 3lu * (1 << 30);
        workspaceSize = 3ll << 30; // 3 GB
        std::cout<<"3------------------------------------"<< std::endl;
        // std::cout<<workspaceSize/1024/1024/1024<< std::endl;
        // std::cout<<std::bitset<64>(workspaceSize)<< std::endl;
        workspace = alloc(workspaceSize);
    }
    virtual ~ASCENDRuntimeObj() {
        dealloc(workspace);
        aclrtDestroyStream(stream);
        aclrtDestroyContext(aclnn);
	aclrtResetDevice(deviceId);
	aclFinalize();
    }
    string toString() const override;

    void run(const Graph &graph, bool tune = false,
             bool profiling = false) const;
    // double runEvaluation(const Graph &graph, int nWarmups,
    //                      int nEvaluations) const;
    void sync() const;
    ASCENDPtr alloc(size_t size) override {
        void *ptr;
        checkASCENDError(
            aclrtMalloc((void **)&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
        return ptr;
    }
    void dealloc(void *ptr) override { aclrtFree(ptr); }
    aclrtContext *ASCENDHandle() const { return nullptr; }
    ASCENDPtr getWorkspace(size_t size) const {
        IT_ASSERT(size <= workspaceSize);
        return workspace;
    }

    void copyBlobFromCPU(void *dst, const void *src,
                         size_t bytes) const override {
        aclrtMemcpy(dst, 1024*1024*1024, const_cast<void *>(src), bytes,
                   ACL_MEMCPY_HOST_TO_DEVICE);
    }

    void copyBlobToCPU(void *dst, const void *src,
                       size_t bytes) const override {
        aclrtMemcpy(dst, 1024*1024*1024, const_cast<void *>(src), bytes,
                   ACL_MEMCPY_DEVICE_TO_HOST);
    }

    void copyBlobInsideRuntime(void *dst, const void *src,
                               size_t bytes) const override {
        aclrtMemcpy(dst, 1024*1024*1024, const_cast<void *>(src), bytes,
                   ACL_MEMCPY_DEVICE_TO_DEVICE);
    }

    void initComm(const string &, int, int) override { IT_TODO_HALT(); }

    CommunicatorObj &getCommunicator() const override { IT_TODO_HALT(); }

  private:
    void runWithoutSync(const Graph &graph, bool tune, bool profiling) const;
};

} // namespace infini

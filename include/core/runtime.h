#pragma once
#include "core/common.h"
#include "core/communicator.h"
#include "core/op_type.h"
#include "core/ref.h"
#include <infiniop/handle.h>
#include <memory>

namespace infini {

/***************** Forward declaration begin *****************/
class TensorBaseObj;
class TensorObj;
class OperatorObj;
class GraphObj;
class GraphHandlerObj;
class RuntimeObj;
class BlobObj;
class HandleObj;
template <typename T> class WorkspaceObj;

using TensorBase = Ref<TensorBaseObj>;
using Tensor = Ref<TensorObj>;
using Operator = Ref<OperatorObj>;
using Graph = Ref<GraphObj>;
using GraphHandler = Ref<GraphHandlerObj>;
using Runtime = Ref<RuntimeObj>;
using Blob = Ref<BlobObj>;
using Handle = Ref<HandleObj>;
template <typename T> using Workspace = Ref<WorkspaceObj<T>>;

using TensorVec = vector<Tensor>;
using OpVec = vector<Operator>;
using OpLists = list<Operator>;

using VType = uint32_t;

enum class Device {
    CPU = 0,
    CUDA,
    BANG,
    ASCEND,
    METAX,
    MOORE,
    ILUVATAR,
    KUNLUN,
    SUGON,
};
/***************** Forward declaration end *****************/

struct HandleObj {
    Device device;
    int deviceId;
};
class RuntimeObj : public std::enable_shared_from_this<RuntimeObj> {
  protected:
    Handle handle;

  public:
    explicit RuntimeObj(Handle h) : handle(h) {}
    RuntimeObj(Device device, int deviceId = 0)
        : handle(make_ref<HandleObj>()) {
        handle->device = device;
        handle->deviceId = deviceId;
    }
    RuntimeObj(RuntimeObj &other) = delete;
    RuntimeObj &operator=(RuntimeObj const &) = delete;
    virtual ~RuntimeObj() {}

    /**
     * @brief Execute a graph.
     *
     * @param graph
     * @param tune If there is no performance record, whether to tune it. These
     * can be independent method.
     * @param profiling Whether to print breakdown of time
     */
    virtual void run(const Graph &graph, bool tune = false,
                     bool profiling = false) const = 0;
    virtual void *alloc(size_t size) = 0;
    virtual void dealloc(void *ptr) = 0;
    /**
     * @brief Get the execution time of each operator in performance record. No
     * execution happens.
     *
     * @param graph
     * @param profiling Whether to print breakdown of time
     * @return double Return the sum of perf time for each operator
     */
    double getPerfTime(const Graph &graph, bool profiling = false) const;
    Blob allocBlob(size_t size);
    bool isCpu() const { return handle->device == Device::CPU; }
    bool isCuda() const { return handle->device == Device::CUDA; }
    bool isBang() const { return handle->device == Device::BANG; }
    bool isKUNLUN() const { return handle->device == Device::KUNLUN; }
    bool isAscend() const { return handle->device == Device::ASCEND; }
    void copyBlob(const TensorObj *dst, const TensorObj *src) const;
    // TODO: unify these copy APIs
    virtual void copyBlobFromCPU(void *dst, const void *src,
                                 size_t bytes) const = 0;
    virtual void copyBlobToCPU(void *dst, const void *src,
                               size_t bytes) const = 0;
    virtual string toString() const = 0;

    int getDeviceId() const { return handle->deviceId; }

    virtual void initComm(const string &name, int worldSize, int rank) = 0;

    virtual CommunicatorObj &getCommunicator() const = 0;
    virtual size_t getWorkspaceSize() const = 0;
    virtual void *getCurrentStream() const { return nullptr; }
    virtual void *getWorkspace(size_t size) const = 0;

    // InfiniopHandle getInfiniHandle() const {
    //     // InfiniopHandle infinihandle;
    //     // infinihandle.device = handle->device;
    //     // infinihandle.deviceId = handle->deviceId;
    //     // return infinihandle;
    //     return {handle->device, handle->deviceId};
    // }

  protected:
    void printProfilingData(double totTime,
                            const std::map<OpType, double> &opTime,
                            const std::map<OpType, int> &opCnt) const;
    virtual void copyBlobInsideRuntime(void *dst, const void *src,
                                       size_t bytes) const = 0;
};

class CpuRuntimeObj : public RuntimeObj {
  public:
    CpuRuntimeObj(Device dev) : RuntimeObj(dev, 0) {}

    void run(const Graph &graph, bool tune = false,
             bool profiling = false) const override;

    void copyBlobFromCPU(void *dst, const void *src,
                         size_t bytes) const override;
    void copyBlobToCPU(void *dst, const void *src, size_t bytes) const override;
    void copyBlobInsideRuntime(void *dst, const void *src,
                               size_t bytes) const override;
    void initComm(const string &, int, int) override { IT_TODO_HALT(); }

    CommunicatorObj &getCommunicator() const override { IT_TODO_HALT(); }
};

class NativeCpuRuntimeObj : public CpuRuntimeObj {
  private:
    size_t workspaceSize;
    void *workspace;

  public:
    NativeCpuRuntimeObj() : CpuRuntimeObj(Device::CPU) {
        workspaceSize = 7ll << 30; // 7 GB
        workspace = alloc(workspaceSize);
    }

    static Ref<NativeCpuRuntimeObj> &getInstance() {
        static Ref<NativeCpuRuntimeObj> instance =
            make_ref<NativeCpuRuntimeObj>();
        return instance;
    }
    void dealloc(void *ptr) override { return free(ptr); };

    void *alloc(size_t size) override {
        return calloc((size + sizeof(uint64_t) - 1) / sizeof(uint64_t),
                      sizeof(uint64_t));
    };
    string toString() const override;
    size_t getWorkspaceSize() const override { return workspaceSize; }

    void *getWorkspace(size_t size) const override {
        IT_ASSERT(size <= workspaceSize);
        return workspace;
    }
};

} // namespace infini

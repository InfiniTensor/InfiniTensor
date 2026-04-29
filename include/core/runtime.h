#pragma once
#include "core/common.h"
#include "core/communicator.h"
#include "core/op_type.h"
#include "core/ref.h"
#include "device.h"
#include "handle.h"
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
template <typename T> class WorkspaceObj;

using TensorBase = Ref<TensorBaseObj>;
using Tensor = Ref<TensorObj>;
using Operator = Ref<OperatorObj>;
using Graph = Ref<GraphObj>;
using GraphHandler = Ref<GraphHandlerObj>;
using Runtime = Ref<RuntimeObj>;
using Blob = Ref<BlobObj>;
template <typename T> using Workspace = Ref<WorkspaceObj<T>>;

using TensorVec = vector<Tensor>;
using OpVec = vector<Operator>;
using OpLists = list<Operator>;

using VType = uint32_t;

// Use InfiniOps Device as the unified device abstraction.
// InfiniOps Device is a class with Type enum (kCpu, kNvidia, kCambricon, etc.)
// and an index field for multi-device support.
using Device = infini::ops::Device;
/***************** Forward declaration end *****************/

class RuntimeObj : public std::enable_shared_from_this<RuntimeObj> {
  protected:
    Device device;
    int deviceId;

    // Workspace management
    static constexpr size_t kDefaultWorkspaceSize = 7ull << 30;
    size_t workspaceSize_ = kDefaultWorkspaceSize;
    size_t workspaceCursor_ = 0;
    void *workspace_ = nullptr;

  public:
    explicit RuntimeObj(Device device, int deviceId = 0);
    RuntimeObj(RuntimeObj &other) = delete;
    RuntimeObj &operator=(RuntimeObj const &) = delete;
    ~RuntimeObj();

    /**
     * @brief Execute a graph.
     */
    void run(const Graph &graph, bool tune = false,
             bool profiling = false) const;

    void *alloc(size_t size);
    void dealloc(void *ptr);

    /**
     * @brief Get the execution time of each operator in performance record. No
     * execution happens.
     */
    double getPerfTime(const Graph &graph, bool profiling = false) const;
    Blob allocBlob(size_t size);

    bool isCpu() const {
        return device.type() == Device::Type::kCpu;
    }
    bool isCuda() const {
        return device.type() == Device::Type::kNvidia;
    }
    bool isBang() const {
        return device.type() == Device::Type::kCambricon;
    }
    bool isKUNLUN() const {
        return device.type() == Device::Type::kKunlun;
    }
    bool isAscend() const {
        return device.type() == Device::Type::kAscend;
    }

    void copyBlob(const TensorObj *dst, const TensorObj *src) const;
    void copyBlobFromCPU(void *dst, const void *src, size_t bytes) const;
    void copyBlobToCPU(void *dst, const void *src, size_t bytes) const;

    int getDeviceId() const { return deviceId; }
    const Device &getDevice() const { return device; }

    // Workspace management for operators that need scratch memory.
    void *getWorkspace(size_t size);
    void resetWorkspace();

    // Create a populated Handle for InfiniOps operator calls.
    infini::ops::Handle makeHandle() const;

    string toString() const;

    virtual void initComm(const string &name, int worldSize, int rank) {
        IT_TODO_HALT();
    }

    virtual CommunicatorObj &getCommunicator() const { IT_TODO_HALT(); }

    // Internal constructor that skips workspace allocation
    // (for temporary helper runtimes that don't need workspace).
    struct NoWorkspace {};
    RuntimeObj(Device device, int deviceId, NoWorkspace)
        : device(device), deviceId(deviceId) {}

  protected:
    void printProfilingData(double totTime,
                            const std::map<OpType, double> &opTime,
                            const std::map<OpType, int> &opCnt) const;
    void copyBlobInsideRuntime(void *dst, const void *src,
                               size_t bytes) const;
};

} // namespace infini

#pragma once
#include "core/common.h"
#include "core/communicator.h"
#include "core/op_type.h"
#include "core/ref.h"
#include "device.h"
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

  public:
    explicit RuntimeObj(Device device, int deviceId = 0)
        : device(device), deviceId(deviceId) {}
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
    void run(const Graph &graph, bool tune = false,
             bool profiling = false) const;
    virtual void *alloc(size_t size) {
        return calloc((size + sizeof(uint64_t) - 1) / sizeof(uint64_t),
                      sizeof(uint64_t));
    }
    virtual void dealloc(void *ptr) { return free(ptr); }
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
    // TODO: unify these copy APIs
    virtual void copyBlobFromCPU(void *dst, const void *src,
                                 size_t bytes) const;
    virtual void copyBlobToCPU(void *dst, const void *src,
                               size_t bytes) const;
    virtual string toString() const { return "Runtime"; }

    int getDeviceId() const { return deviceId; }

    const Device &getDevice() const { return device; }

    virtual void initComm(const string &name, int worldSize, int rank) {
        IT_TODO_HALT();
    }

    virtual CommunicatorObj &getCommunicator() const { IT_TODO_HALT(); }

  protected:
    void printProfilingData(double totTime,
                            const std::map<OpType, double> &opTime,
                            const std::map<OpType, int> &opCnt) const;
    virtual void copyBlobInsideRuntime(void *dst, const void *src,
                                       size_t bytes) const;
};

} // namespace infini
